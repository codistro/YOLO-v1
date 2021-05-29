import torch

def intersection_over_union(bbox_pred, bbox, type='center'):

    if type == 'center':

        #calculations for box1

        box1_x1, box1_y1, w1, h1 = [float(_) for _ in bbox_pred]

        box1_x1 -= w1/2
        box1_y1 -= h1/2

        box1_x2 = box1_x1 + w1
        box1_y2 = box1_y1 + h1

        box1_area = w1 * h1


        # calculations for box2

        box2_x1, box2_y1, w1, h1 = [float(_) for _ in bbox]

        box2_x1 -= w1 / 2
        box2_y1 -= h1 / 2

        box2_x2 = box2_x1 + w1
        box2_y2 = box2_y1 + h1

        box2_area = w1 * h1

        #calculations for intersection box

        intersection_x1 = max(box1_x1, box2_x1)
        intersection_y1 = max(box1_y1, box2_y1)
        intersection_x2 = min(box1_x2, box2_x2)
        intersection_y2 = min(box1_y2, box2_y2)

        intersection_area = intersection_x2 - intersection_x1 * intersection_y2 - intersection_y1

        IoU = (intersection_area) / (box1_area + box2_area - intersection_area)

        print(IoU)



