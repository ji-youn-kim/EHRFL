import os
import torch
from torch import mean
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from torch.nn.parallel import DistributedDataParallel

import utils
from .base import Trainer
from model import GenHPF

logger = get_logger(__name__, "INFO")

class BaseTrainer(Trainer):
    
    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)
        
        self.model = GenHPF(self.args)
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

        self.train_loader = self.dataloader_set(
            torch.utils.data.ConcatDataset(self.datasets["train"].values()),
            self.args.batch_size,
            collator=list(self.datasets["train"].values())[0].collator,
        )
        if not args.extract_latent:
            if 'valid' in self.data_types:
                self.valid_loaders = dict()
                for data_name in self.datasets["valid"].keys(): 
                    self.valid_loaders[data_name] = self.accelerator.prepare(
                            self.dataloader_set(
                            self.datasets['valid'][data_name],
                            self.args.batch_size,
                            collator=self.datasets['valid'][data_name].collator,
                        )
                    )
            if 'test' in self.data_types:
                self.test_loaders = dict()
                for data_name in self.datasets["test"].keys():
                    self.test_loaders[data_name] = self.accelerator.prepare(
                            self.dataloader_set(
                            self.datasets['test'][data_name],
                            self.args.batch_size,
                            collator=self.datasets['test'][data_name].collator,
                        )
                    )
        
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        self.accelerator.free_memory()
        
    def train(self):
        if not self.args.extract_latent:
            for n_epoch in range(self.args.communications):
            
                logger.info(f'[Epoch] {n_epoch}')
                self.model.train()

                for n_iter, sample in tqdm(enumerate(self.train_loader), disable=not self.accelerator.is_main_process):
                    self.optimizer.zero_grad()
                    args_to_pass = {**(vars(self.args)), 'n_iter':n_iter, 'data_type':'train'}
                    output = self.model(**sample["net_input"], **args_to_pass)

                    if isinstance(self.model, DistributedDataParallel):
                        logits = self.model.module.get_logits(output)
                        targets = self.model.module.get_targets(sample)
                    else:
                        logits = self.model.get_logits(output)
                        targets = self.model.get_targets(sample)

                    loss, logging_outputs = self.criterion(logits, targets)

                    self.accelerator.backward(loss)

                    if self.accelerator.mixed_precision == 'fp16' and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                    self.accelerator.wait_for_everyone()

                    for task in self.args.pred_tasks:
                        for key in logging_outputs[task.name].keys():
                            if key == 'loss':
                                logging_outputs[task.name][key] = \
                                    mean(self.accelerator.gather_for_metrics(logging_outputs[task.name][key])).item()
                            elif key in ['score', 'true']:
                                logging_outputs[task.name][key] = \
                                   self.accelerator.gather_for_metrics(logging_outputs[task.name][key]).cpu().detach().numpy()
                            else:
                                raise NotImplementedError("What else?")
                    logging_outputs['loss'] = \
                        mean(self.accelerator.gather_for_metrics(logging_outputs['loss'])).item()
                
                    self.accelerator.wait_for_everyone()

                    self.metric(logging_outputs)

                with torch.no_grad():
                    train_metric_dict = self.metric.get_epoch_dict(len(self.train_loader)) 
            
                log_dict = utils.log_from_dict(
                    self.args, train_metric_dict, "train", self.args.src_data, n_epoch, self.accelerator.is_main_process
                )

                if self.args.debug == False:
                    self.accelerator.log(log_dict)

                self.accelerator.wait_for_everyone()

                if 'valid' in self.data_types:
                    # Validation
                    for c, data_name in enumerate(self.datasets['valid'].keys()):
                        logger.info(f'[Epoch] {n_epoch}: Validation on {data_name}')

                        if not self.valid_stop[data_name]:

                            metric_dict = self.inference(
                                self.args,
                                self.accelerator, 
                                self.model, 
                                self.valid_loaders[data_name], 
                                "valid",
                                data_name,
                                n_epoch,
                                self.criterion,
                                self.metric
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
                                    state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                                    self.accelerator.save(state_dict, best_model_path)

                            if self.early_stopping_dict[data_name].early_stop:
                                logger.info(f"data_name: {data_name}, Early stopping!")
                                self.valid_stop[data_name] = 1

                            self.accelerator.wait_for_everyone()

                    if sum(self.valid_stop.values()) == len(self.datasets["valid"]):
                        logger.info(f"All valid finished at {n_epoch}")
                        if self.args.debug == False:
                            self.accelerator.log({"all_clients_stop": n_epoch})
                        break

        if 'test' in self.data_types or self.args.extract_latent:
            # Test
            for c, data_name in enumerate(self.datasets["train"].keys()):

                if not self.args.extract_latent:
                    best_model_path = os.path.join(
                        self.args.save_dir,
                        self.args.exp_name,
                        self.args.save_prefix + f"_{data_name}_best.pt",
                    )
                else:
                    # load pretrained model
                    if ("mimiciii" in self.args.exp_name) or ("eicu" in self.args.exp_name):
                        pretrained_data = "_".join(self.args.exp_name.split("_")[-2:])
                    elif ("mimiciv" in self.args.exp_name):
                        pretrained_data = self.args.exp_name.split("_")[-1]
                    best_model_path = os.path.join(
                        self.args.save_dir,
                        self.args.exp_name,
                        self.args.save_prefix + f"_{pretrained_data}_best.pt",
                    )
                self.accelerator.print(f"Loaded best model from {best_model_path}")
                state_dict = torch.load(best_model_path, map_location="cpu")
                
                model = GenHPF(self.args)
                model.load_state_dict(state_dict, strict=True)
                model = self.accelerator.prepare(model)
                self.accelerator.free_memory()

                if not self.args.extract_latent:
                    metric_dict = self.inference(
                        self.args,
                        self.accelerator, 
                        model, 
                        self.test_loaders[data_name], 
                        "test",
                        data_name,
                        n_epoch,
                        self.criterion,
                        self.metric
                    )

                    logger.info(f"Test finished at epoch {n_epoch}")
                    
                else:
                    metric_dict = self.inference(
                    self.args,
                    self.accelerator, 
                    model, 
                    self.train_loader, 
                    "test",
                    data_name,
                    300,
                    self.criterion,
                    self.metric
                )

            

