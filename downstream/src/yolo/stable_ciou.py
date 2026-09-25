"""Compute regression CIoU geometry in FP32 within the YOLO box loss."""
import torch
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

def fp32_ciou(box1, box2):
    with torch.autocast(device_type=box1.device.type, enabled=False):
        return bbox_iou(box1.float(), box2.float(), xywh=False, CIoU=True)

class FP32CIoUBboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = fp32_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

def install(model):
    if not hasattr(model, 'criterion'):
        model.criterion = model.init_criterion()
    model.criterion.bbox_loss = FP32CIoUBboxLoss(model.model[-1].reg_max).to(next(model.parameters()).device)
