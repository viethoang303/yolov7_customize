from typing import List
from sklearn.cluster import DBSCAN
from collections import Counter
# from utils.general import bbox_iou
import numpy as np
import torch
import math

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return (1 - iou + rho2 / c2).numpy()  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return (1 - iou + (rho2 / c2 + v * alpha)).numpy()  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return (1 - iou + (c_area - union) / c_area).numpy()  # GIoU
    else:
        return 1 - iou  # IoU

def get_points(boxes, size) -> List:
    points = []
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        try: x0 = x0.cpu()
        except: x0 = float(x0)
        try: y0 = y0.cpu()
        except: y0 = float(y0)
        try: x1 = x1.cpu()
        except: x1 = float(x1)
        try: y1 = y1.cpu()
        except: y1 = float(y1)
        x0 = max(0, x0)
        y0 = max(0, y0)
        w = x1 - x0
        h = y1 - y0
        # points.append([(x0 + w / 2)/size, (y0 + h / 2)/size])
        points.append([x0/size,y0/size,x1/size,y1/size])
    return points

def dectect_event_by_cluster_crowd(boxes, distance_object=0.1, min_object=4, size=1280):
    if boxes is not None and len(boxes) > 1:
        points = get_points(boxes, size) # sua o day
        cluster_dict = {}
        db = DBSCAN(eps=distance_object, min_samples=min_object, metric=bbox_iou).fit(points)
        label_clusters = db.labels_
        # print(label_clusters)
        clusters_list = Counter(label_clusters)
        for i in range(0, max(clusters_list) + 1):
            if clusters_list[i] < min_object:
                continue
            else:
                cluster_dict.setdefault(i, clusters_list[i])
        boxes_in_cluster = get_points_in_cluster(boxes, cluster_dict, label_clusters)
    else:
        boxes_in_cluster = []
    return boxes_in_cluster
    
def get_points_in_cluster(boxes, cluster_dict, label_clusters): 
    boxes_in_cluster = []
    for cluster in cluster_dict.keys():
        boxes_cluster = []
        for i in range(0, len(label_clusters)):
            if label_clusters[i] == int(cluster):
                boxes_cluster.append(boxes[i])
            else:
                continue
        print(len(boxes_cluster))
        boxes_in_cluster.append(boxes_cluster)
    return boxes_in_cluster

# def draw(boxes_in_cluster):
#     for i in range(len(boxes_in_cluster)):
#         x_min, y_min, x_max, y_max = np.Inf, np.Inf, 0, 0
#         for j in range(len(i)):
