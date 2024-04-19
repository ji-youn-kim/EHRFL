import os
from copy import deepcopy
from accelerate.logging import get_logger
import torch

from .base import Trainer
from model import GenHPF
from local_training import train_fedprox, train_naive
from communication_func import communication

logger = get_logger(__name__, "INFO")

class FedTrainer(Trainer):
    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)

        self.client_weights = dict()
        for data in self.args.src_data:
            self.client_weights[data] = len(self.datasets["train"][data])
        tot_len = sum(self.client_weights.values())
        for data in self.client_weights.keys():
            self.client_weights[data] = self.client_weights[data] / tot_len

        self.server_model = GenHPF(self.args)
        self.local_models = dict()
        for data in self.args.src_data:
            self.local_models[data] = deepcopy(self.server_model)
        self.param_name = [name for name, _ in self.server_model.named_parameters()]
        
        # datasets
        self.train_loaders = dict()
        for data_name in self.datasets["train"].keys():
            self.train_loaders[data_name] = self.dataloader_set(
                self.datasets["train"][data_name],
                self.args.batch_size,
                self.datasets["train"][data_name].collator
            )
        if 'valid' in self.data_types:
            self.valid_loaders = dict()
            for data_name in self.datasets["valid"].keys():
                self.valid_loaders[data_name] = self.dataloader_set(
                    self.datasets["valid"][data_name],
                    self.args.batch_size,
                    self.datasets["valid"][data_name].collator
                )
        if 'test' in self.data_types:
            self.test_loaders = dict()
            for data_name in self.datasets["test"].keys():
                self.test_loaders[data_name] = self.dataloader_set(
                    self.datasets["test"][data_name],
                    self.args.batch_size,
                    self.datasets["test"][data_name].collator
                )

    def train(self):
        for comms in range(self.args.communications):
            logger.info(f"[Comms] {comms}")

            # Train
            for data_name in self.datasets['train'].keys():
                self.accelerator.print(f"Start training {data_name}")
                model, train_loader = (
                    self.local_models[data_name],
                    self.train_loaders[data_name]
                )
                if self.args.optimizer == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
                elif self.args.optimizer == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
                model, optimizer, train_loader = self.accelerator.prepare(
                    model, optimizer, train_loader
                )
                self.accelerator.free_memory()
                model.train()
                logger.info(f"[Comms] {comms}: Training on {data_name}")

                for epoch in range(self.args.local_epochs):
                    total_steps = epoch + comms + self.args.local_epochs

                    if (
                        self.args.algorithm == "fedprox"
                        or self.args.algorithm == "fedpxn"
                    ) and comms > 0:
                        train_fedprox(
                            self.args,
                            self.accelerator,
                            self.server_model,
                            model,
                            optimizer,
                            self.criterion,
                            total_steps,
                            train_loader,
                            self.metric,
                            data_name,
                            self.param_name,
                        )
                    else:  # FedAvg, FedBN
                        train_naive(
                            self.args,
                            self.accelerator,
                            model,
                            optimizer,
                            self.criterion,
                            total_steps,
                            train_loader,
                            self.metric,
                            data_name,
                        )

                self.local_models[data_name] = self.accelerator.unwrap_model(model).to("cpu")

                del model, optimizer, train_loader
                self.accelerator.wait_for_everyone()

            logger.info("Start Communication")

            if self.accelerator.is_main_process:
                self.server_model, self.local_models = communication(
                    self.args, self.server_model, self.local_models, self.client_weights
                )

            logger.info("Done Communication")
            
            self.server_model = [self.server_model]
            tmp_local_models = [i for i in self.local_models.values()]
            torch.distributed.broadcast_object_list(tmp_local_models, 0)
            torch.distributed.broadcast_object_list(self.server_model, 0)
            self.server_model = self.server_model[0]
            for i, data_name in enumerate(self.local_models.keys()):
                self.local_models[data_name] = tmp_local_models[i]
            del tmp_local_models

            total_steps = (comms + 1) * self.args.local_epochs

            self.accelerator.wait_for_everyone()

            # Validation
            if 'valid' in self.data_types:
                for data_name in self.datasets['valid'].keys():
                    logger.info(f"[Comms] {comms}: Validation on {data_name}")
                    
                    if not self.valid_stop[data_name]:
                        model, valid_loader = (
                            self.local_models[data_name], 
                            self.valid_loaders[data_name]
                        )
                        model, valid_loader = self.accelerator.prepare(
                            model, valid_loader
                        )
                        self.accelerator.free_memory()

                        metric_dict = self.inference(
                            self.args,
                            self.accelerator,
                            model,
                            valid_loader,
                            "valid",
                            data_name,
                            total_steps,
                            self.criterion,
                            self.metric,
                        )

                        if self.early_stopping_dict[data_name](
                            metric_dict[self.metric.update_target]
                        ):
                            if self.accelerator.is_main_process:
                                best_model_path = os.path.join(
                                    self.args.save_dir,
                                    self.args.exp_name,
                                    self.args.save_prefix + f"_{data_name}_best.pt",
                                )
                                torch.save(self.local_models[data_name].state_dict(), best_model_path)
                                if self.args.debug == False:
                                    self.accelerator.log({f"{data_name}_best": comms})
                        if self.early_stopping_dict[data_name].early_stop:
                            logger.info(f"data_name : {data_name}, Early stopping!")
                            self.valid_stop[data_name] = 1

                        del model, valid_loader
                        self.accelerator.wait_for_everyone()

                    if self.args.debug == False:
                        self.accelerator.log({"comms": comms})
                    
                if sum(self.valid_stop.values()) == len(self.datasets['valid']):
                    logger.info(f"All valid finished at {comms}")
                    if self.args.debug == False:
                        self.accelerator.log({"all_clients_stop": comms})
                    break
        
        if 'test' in self.data_types:
            # Test
            for data_name in self.datasets["test"].keys():
                self.local_models[data_name].load_state_dict(
                    torch.load(
                        os.path.join(
                            self.args.save_dir,
                            self.args.exp_name,
                            self.args.save_prefix + f"_{data_name}_best.pt",
                        ),
                        map_location="cpu",
                    )
                )
            
            for data_name in self.datasets["test"].keys():
                model, test_loader = (
                    self.local_models[data_name],
                    self.test_loaders[data_name],
                )
                model, test_loader = self.accelerator.prepare(
                    model, test_loader
                )
                self.accelerator.free_memory()

                metric_dict = self.inference(
                    self.args,
                    self.accelerator,
                    model,
                    test_loader,
                    "test",
                    data_name,
                    total_steps,
                    self.criterion,
                    self.metric,
                )

                del model, test_loader
        if self.args.debug == False:
            self.accelerator.log({"comms": -1})
                