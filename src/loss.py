import torch
import torch.nn as nn

class PredLoss:
    def __init__(self, args):
        self.args = args
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def __call__(self, logits, targets):
        
        losses = dict()
        logging_output = dict()

        for task in self.args.pred_tasks:
            pred = logits[task.name]
            target = targets[task.name]
            if task.property == "binary":
                pred = pred.view(-1)
                target = target.view(-1)
                loss = self.bce(input=pred, target=target)

            elif task.property == "multi-label":
                loss = self.bce(input=pred, target=target)
            
            elif task.property == "multi-class":
                loss = self.ce(input=pred, target=target)

            losses[task.name] = loss 

            with torch.no_grad():
                logging_output[task.name] = {
                    "loss": loss.detach(),
                    "score": torch.sigmoid(pred).detach(),
                    "true": target.detach()
                }

        total_loss = sum(list(losses.values())) 
        logging_output["loss"] = total_loss

        return total_loss, logging_output
