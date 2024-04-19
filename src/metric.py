from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


class PredMetric:
    def __init__(self, args, target="auroc"):
        self.args = args
        self._update_target = target
        
        self.reset()

    def getter(self):
        return self.metric

    def reset(self):
        self.metric = dict()
        for task in self.args.pred_tasks:
            self.metric[task.name] = {
                "loss": 0,
                "score": [],
                "true": []
            }
        self.metric["loss"] = 0

    def __call__(self, logging_outputs):
        for task in self.args.pred_tasks:
            self.metric[task.name]["loss"] += logging_outputs[task.name]["loss"]
            self.metric[task.name]["score"].append(logging_outputs[task.name]["score"]) 
            self.metric[task.name]["true"].append(logging_outputs[task.name]["true"])
        self.metric["loss"] += logging_outputs["loss"]

    def get_epoch_dict(self, total_iter):
        self.epoch_dict = dict()
        total_auroc = 0
        for task in self.args.pred_tasks:
            self.epoch_dict[task.name] = {
                "auroc": self.auroc(task),
                "auprc": self.auprc(task),
                "loss": self.metric[task.name]["loss"] / total_iter,
            }
            total_auroc += self.macro_auroc(task)
        self.epoch_dict["loss"] = self.metric["loss"] / total_iter
        self.epoch_dict["auroc"] = total_auroc / (len(self.args.pred_tasks))

        self.reset()

        return self.epoch_dict

    @property
    def compare(self):
        return "decrease" if self.update_target == "loss" else "increase"

    @property
    def update_target(self):
        return self._update_target

    def auroc(self, task):
        y_true = np.concatenate(self.metric[task.name]["true"])
        y_score = np.concatenate(self.metric[task.name]["score"])
        try:
            if task.property in ["multi-class", "multi-label"]:
                missing = np.where(np.sum(y_true, axis=0) == 0)[0]
                y_true = np.delete(y_true, missing, 1)
                y_score = np.delete(y_score, missing, 1)
            return roc_auc_score(y_true=y_true, y_score=y_score, average="micro", multi_class="ovr")
        except:
            return float("nan")

    def macro_auroc(self, task):
        y_true = np.concatenate(self.metric[task.name]["true"])
        y_score = np.concatenate(self.metric[task.name]["score"])
        try:
            if task.property in ["multi-class", "multi-label"]:
                missing = np.where(np.sum(y_true, axis=0) == 0)[0]
                y_true = np.delete(y_true, missing, 1)
                y_score = np.delete(y_score, missing, 1)
            return roc_auc_score(y_true=y_true, y_score=y_score, average="macro")
        except:
            return float("nan")

    def auprc(self, task):
        y_true = np.concatenate(self.metric[task.name]["true"])
        y_score = np.concatenate(self.metric[task.name]["score"])
        try:
            return average_precision_score(y_true=y_true, y_score=y_score, average="micro")
        except:
            return float("nan")
