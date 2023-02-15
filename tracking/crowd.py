from typing import List, Optional

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

