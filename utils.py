import torch


def intersection_over_union(bbox_pred, bbox, type='center'):
    if type == 'center':
        # calculations for box1

        box1_x1, box1_y1, w, h = bbox_pred[..., 0], bbox_pred[..., 1], bbox_pred[..., 2], bbox_pred[..., 3]

        box1_x1 = box1_x1 - (w / 2)
        box1_y1 = box1_y1 - (h / 2)

        box1_x2 = box1_x1 + w
        box1_y2 = box1_y1 + h

        box1_area = w * h

        # calculations for box2

        box2_x1, box2_y1, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]

        box2_x1 = box2_x1 - w / 2
        box2_y1 = box2_y1 - h / 2

        box2_x2 = box2_x1 + w
        box2_y2 = box2_y1 + h

        box2_area = w * h

        # calculations for intersection box

        intersection_x1 = torch.max(box1_x1, box2_x1)
        intersection_y1 = torch.max(box1_y1, box2_y1)
        intersection_x2 = torch.max(box1_x2, box2_x2)
        intersection_y2 = torch.max(box1_y2, box2_y2)

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        iou = intersection_area / (box1_area + box2_area - intersection_area)

        return iou
