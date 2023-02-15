import collections
from typing import List, Optional
from sklearn import DBSCAN
import numpy as np


class Crowd:
    track_id: Optional[int]
    init_time: float
    bounding_box: np.ndarray  # x1, y1, x2, y2
    feature_vector: Optional[np.ndarray]
    nb_of_persons: int
    area_id: Optional[str]
    image_url: Optional[str]
    video_url: Optional[str]

    def __init__(self, bounding_box: np.ndarray, nb_of_persons: int = 0):
        self.bounding_box = bounding_box
        self.nb_of_persons = nb_of_persons
        self.feature_vector = None
        self.track_id = None
        self.image_url = None
        self.video_url = None
        self.area_id = ""




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
        points.append([x0/size,y0/size,x1/size,y1/size])
    return points

def get_crowds_by_person_boxes(boxes, distance_object=0.1, min_person=5) -> List[Crowd]:
    if boxes is None or len(boxes) < 1:
        return []

    center_points = get_center_points(boxes=boxes)
    crowds: List[Crowd] = []
    db = DBSCAN(eps=distance_object, min_samples=min_person).fit(center_points)
    label_clusters = db.labels_
    n_clusters = len(set(label_clusters)) - (1 if -1 in label_clusters else 0)
    logger.info(f"nb of cluster: {n_clusters}")
    for i in range(n_clusters):
        cluster_boxes = [box for (box, label) in zip(boxes, label_clusters) if label == i]
        cluster_boxes = np.array(cluster_boxes)
        x_min = max(min(cluster_boxes[:, 0]), 0)
        y_min = max(min(cluster_boxes[:, 1]), 0)
        x_max = max(cluster_boxes[:, 2])
        y_max = max(cluster_boxes[:, 3])
        crowds.append(Crowd(bounding_box=np.array([x_min, y_min, x_max, y_max]), nb_of_persons=len(cluster_boxes)))
    return crowds
