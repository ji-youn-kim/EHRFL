from accelerate.logging import get_logger
from dataclasses import dataclass

logger = get_logger(__name__, "INFO")

@dataclass
class Task:
    name: str
    num_classes: int
    property: str


def get_task(pred_task):

    return {
        'mortality': Task('mortality', 1, 'binary'),
        'long_term_mortality': Task('long_term_mortality', 1, 'binary'), 
        'los_3day': Task('los_3day', 1, 'binary'), 
        'los_7day': Task('los_7day', 1, 'binary'),
        'readmission': Task('readmission', 1, 'binary'),
        'final_acuity': Task('final_acuity', 6, 'multi-class'), 
        'imminent_discharge': Task('imminent_discharge', 6, 'multi-class'), 
        'diagnosis': Task('diagnosis', 17, 'multi-label'), 
        'creatinine': Task('creatinine', 5, 'multi-class'), 
        'bilirubin': Task('bilirubin', 5, 'multi-class'), 
        'platelets': Task('platelets', 5, 'multi-class'),
        'wbc': Task('wbc', 3, 'multi-class'),
    }[pred_task]


def log_from_dict(args, metric_dict, data_type, data_name, n_epoch, is_master=False):
    if isinstance(data_name, list):
        data_name = "_".join(data_name)
    log_dict = {"epoch": n_epoch}

    log_dict[data_type + "/" + data_name + "_" + "loss"] = metric_dict["loss"]
    log_dict[data_type + "/" + data_name + "_" + "auroc"] = metric_dict["auroc"]
    if is_master:
        logger.info(
            data_type + "/" + data_name + "_" + "loss" + " : {:.3f}".format(metric_dict["loss"])
        )
        logger.info(
            data_type + "/" + data_name + "_" + "auroc" + " : {:.3f}".format(metric_dict["auroc"])
        )

    for task in args.pred_tasks:
        for metric, values in metric_dict[task.name].items():
            log_dict[data_type + "/" + data_name + "_" + task.name + "_" + metric] = values
            if is_master:
                logger.info(
                    data_type + "/" + data_name + "_" + task.name + "_" + metric + " : {:.3f}".format(values)
                )
                
    return log_dict


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=True, delta=0, compare="increase", metric="auprc"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metric_min = 0
        self.delta = delta
        self.compare_score = self.increase if compare == "increase" else self.decrease
        self.metric = metric

    def __call__(self, target_metric):
        update_token = False
        score = target_metric

        if self.best_score is None:
            self.best_score = score

        if self.compare_score(score):
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                logger.info(
                    f"Validation {self.metric} {self.compare_score.__name__}d {self.target_metric_min:.6f} --> {target_metric:.6f})"
                )
            self.target_metric_min = target_metric
            self.counter = 0
            update_token = True

        return update_token

    def increase(self, score):
        if score < self.best_score + self.delta:
            return True
        else:
            return False

    def decrease(self, score):
        if score > self.best_score + self.delta:
            return True
        else:
            return False