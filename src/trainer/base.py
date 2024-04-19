import torch
from torch import mean
from tqdm.auto import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel

from dataset import HierarchicalEHRDataset
from metric import PredMetric
from loss import PredLoss
import utils


class Trainer:
    def __init__(self, args, accelerator):
        self.args = args
        
        self.accelerator = accelerator

        self.metric = PredMetric(self.args)
        self.criterion = PredLoss(self.args)

        self.datasets = dict()
        self.early_stopping_dict = dict()
        self.data_types = ["train"] + self.args.valid_subsets

        for split in self.data_types:
            self.datasets[split] = OrderedDict()
            for data in self.args.src_data:
                self.datasets[split][data] = self.load_dataset(split, data)

        if 'valid' in self.data_types:
            self.valid_stop = dict()
            for data in self.datasets["valid"].keys():
                self.early_stopping_dict[data] = utils.EarlyStopping(
                    patience=self.args.patience,
                    compare=self.metric.compare,
                    metric=self.metric.update_target,
                )
                self.valid_stop[data] = 0
        
        set_seed(self.args.seed)
    
    def load_dataset(self, split, dataname) -> None:

        return HierarchicalEHRDataset(
            data=dataname,
            split=split,
            args=self.args,
            accelerator=self.accelerator
        )
        
    def dataloader_set(self, dataset, batch_size, collator):
            
        return DataLoader(
            dataset,
            collate_fn=collator,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
    
    def inference(
        self,
        args,
        accelerator,
        model,
        data_loader,
        data_type,
        data_name,
        n_epoch,
        criterion,
        metric,
    ):
        model.eval()
        with torch.no_grad():
            for n_iter, sample in tqdm(enumerate(data_loader), disable=not accelerator.is_main_process):

                args_to_pass = {**(vars(self.args)), 'n_iter':n_iter, 'data': data_name, 'data_type':data_type}
                output = model(**sample["net_input"], **args_to_pass)

                if isinstance(model, DistributedDataParallel):
                    logits = model.module.get_logits(output)
                    targets = model.module.get_targets(sample)
                else:
                    logits = model.get_logits(output)
                    targets = model.get_targets(sample)

                _, logging_outputs = criterion(logits, targets)

                accelerator.wait_for_everyone()

                for task in args.pred_tasks:
                    for key in logging_outputs[task.name].keys():
                        if key == 'loss':
                            logging_outputs[task.name][key] = \
                                mean(accelerator.gather_for_metrics(logging_outputs[task.name][key])).item()
                        elif key in ['score', 'true']:
                            logging_outputs[task.name][key] = \
                                accelerator.gather_for_metrics(logging_outputs[task.name][key]).cpu().detach().numpy()
                        else:
                            raise NotImplementedError("What else?")
                logging_outputs["loss"] = \
                    mean(accelerator.gather_for_metrics(logging_outputs["loss"])).item()

                metric(logging_outputs)  # iter_update

            metric_dict = metric.get_epoch_dict(len(data_loader))

        log_dict = utils.log_from_dict(
            args, metric_dict, data_type, data_name, n_epoch, accelerator.is_main_process
        )

        if args.debug == False:
            accelerator.log(log_dict)

        return metric_dict

    
