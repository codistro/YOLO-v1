import torch
from torch import nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=80):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.coord = 5
        self.noobj = 0.5

        # confidence, x, y, w, h
        self.num_predictions = 5

    def forward(self, pred, target):
        pred = pred.reshape(-1, self.S, self.S, (self.num_predictions * self.B + self.C))

        xy_loss, box_loss, confidence_loss, noobj_confidence_loss, cls_loss = [0] * 5
        box_curr_index = 0

        for i in range(self.B):
            x_pred = pred[..., box_curr_index + 1]
            y_pred = pred[..., box_curr_index + 2]
            w_pred = torch.sqrt(torch.abs(pred[..., box_curr_index + 3]))
            h_pred = torch.sqrt(torch.abs(pred[..., box_curr_index + 4]))

            confidence_target = target[..., box_curr_index]
            x_target = target[..., box_curr_index + 1]
            y_target = target[..., box_curr_index + 2]
            w_target = torch.sqrt(torch.abs(target[..., box_curr_index + 3]))
            h_target = torch.sqrt(torch.abs(target[..., box_curr_index + 4]))

            # (x, y) loss

            xy_loss += torch.sum(
                (torch.square(x_target - x_pred) + torch.square(y_target - y_pred)) * confidence_target)

            # (w, h) box_loss

            box_loss += torch.sum(
                (torch.square(w_target - w_pred) + torch.square(h_target - h_pred)) * confidence_target)

            # confidence_loss
            confidence_pred = intersection_over_union(pred[..., box_curr_index + 1: box_curr_index + 5],
                                                      target[..., box_curr_index + 1: box_curr_index + 5])

            confidence_loss += torch.sum(torch.square(confidence_target - confidence_pred) * confidence_target)

            # noobj confidence_loss

            noobj_confidence_loss += torch.sum(
                torch.square(confidence_target - confidence_pred) * (1 - confidence_target))

            box_curr_index += self.num_predictions

        # class loss

        pred_classes = pred[..., box_curr_index:]
        target_classes = target[..., box_curr_index:]

        cls_loss += torch.sum((torch.square(target_classes) - torch.square(pred_classes)) * target_classes)

        yolo_loss = (self.coord * xy_loss) + (
                self.coord * box_loss) + confidence_loss + (self.noobj * noobj_confidence_loss) + cls_loss

        return yolo_loss
