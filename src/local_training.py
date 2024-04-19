import torch
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from torch import mean

import utils
from torch.nn.parallel import DistributedDataParallel

logger = get_logger(__name__, "INFO")

def train_fedprox(
    args,
    accelerator,
    global_model,
    local_model,
    optimizer,
    criterion,
    n_epoch,
    data_loader,
    metric,
    data_name,
    param_name
):
    logger.info("[Epoch] {}".format(n_epoch))

    for n_iter, sample in tqdm(enumerate(data_loader), disable=not accelerator.is_main_process):
        optimizer.zero_grad()

        args_to_pass = {**(vars(args)), 'n_iter':n_iter, 'data': data_name, 'data_type': 'train'}
        output = local_model(**sample["net_input"], **args_to_pass)

        if isinstance(local_model, DistributedDataParallel):
            logits = local_model.module.get_logits(output)
            targets = local_model.module.get_targets(sample)
        else:
            logits = local_model.get_logits(output)
            targets = local_model.get_targets(sample)

        loss, logging_outputs = criterion(logits, targets)

        if n_iter > 0:
            w_diff = torch.tensor(0.0, device=accelerator.device) 
            for name, w, w_t in zip(
                param_name,
                global_model.parameters(),
                local_model.parameters(),
            ):
                w = w.to(accelerator.device)
                if args.algorithm == "fedprox":
                    if "norm" not in name:
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                else:
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

            loss += args.mu / 2.0 * w_diff

        accelerator.wait_for_everyone()

        accelerator.backward(loss)

        if accelerator.mixed_precision == 'fp16' and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(local_model.parameters(), 1.0)
        
        optimizer.step()

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
        
        accelerator.wait_for_everyone()

        metric(logging_outputs)  # iter_update

    with torch.no_grad():
        train_metric_dict = metric.get_epoch_dict(len(data_loader))

    log_dict = utils.log_from_dict(
        args, train_metric_dict, "train", data_name, n_epoch, accelerator.is_main_process
    )

    if args.debug == False:
        accelerator.log(log_dict)

    return log_dict


def train_naive(
    args,
    accelerator,
    local_model,
    optimizer,
    criterion,
    n_epoch,
    data_loader,
    metric,
    data_name,
):

    logger.info("[Epoch] {}".format(n_epoch))

    for n_iter, sample in tqdm(enumerate(data_loader), disable=not accelerator.is_main_process):
        optimizer.zero_grad()

        args_to_pass = {**(vars(args)), 'n_iter':n_iter, 'data': data_name, 'data_type': 'train'}
        output = local_model(**sample["net_input"], **args_to_pass)

        if isinstance(local_model, DistributedDataParallel):
            logits = local_model.module.get_logits(output)
            targets = local_model.module.get_targets(sample)
        else:
            logits = local_model.get_logits(output)
            targets = local_model.get_targets(sample)

        loss, logging_outputs = criterion(logits, targets)

        accelerator.backward(loss)
        
        if accelerator.mixed_precision == 'fp16' and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(local_model.parameters(), 1.0)

        optimizer.step()

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
        
        accelerator.wait_for_everyone()

        metric(logging_outputs)  # iter_update

    with torch.no_grad():
        train_metric_dict = metric.get_epoch_dict(len(data_loader))

    log_dict = utils.log_from_dict(
        args, train_metric_dict, "train", data_name, n_epoch, accelerator.is_main_process
    )

    if args.debug == False:
        accelerator.log(log_dict)

    return log_dict

