"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import time

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from filterpy.kalman import KalmanFilter

from typing import List
from .crowd import Crowd

np.random.seed(0)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, crowd: Crowd, init_time: int):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(crowd.bounding_box)
        self.time_since_update = 0
        # Track id, set maximum to 10^9. Track id reset after 10^9 records
        self.id: int = KalmanBoxTracker.count % 1000000000
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Added attributes
        self.init_time = init_time
        self.detection_index: int = -1
        self.crowd = crowd
        self.crowd.init_time = self.init_time

    def update(self, crowd: Crowd):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(crowd.bounding_box))
        self.update_crowd_info(crowd=crowd)

    def update_crowd_info(self, crowd):
        self.crowd.bounding_box = crowd.bounding_box
        self.crowd.feature_vector = crowd.feature_vector
        self.crowd.area_id = crowd.area_id

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Sort(object):
    def __init__(self, max_age=5, min_hits=3, max_distance=0.9, recognition_threshold=0.72):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.recognition_threshold = recognition_threshold

    def update(self, crowds: List[Crowd], timestamp=float(time.time() * 1000)) -> List[KalmanBoxTracker]:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame
        even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(crowds=crowds,
                                                                                   trackers=self.trackers,
                                                                                   max_distance=self.max_distance)
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(crowds[m[0]])
            # set detection_index to map with other detection values such as bounding boxes, landmarks, key points ...
            self.trackers[m[1]].detection_index = m[0]
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(crowds[i], timestamp)
            trk.detection_index = i
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk)
            i -= 1

            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                del self.trackers[i]
        # Returned ret contain either empty list or list of KalmanBoxTracker having time_since_update < 1
        return ret


def get_euclid_distance_cost_matrix(crowds: List[Crowd], trackers: List[KalmanBoxTracker], max_distance=100):
    cp_detects = []
    cp_tracks = []

    for crowd in crowds:
        bbox = crowd.bounding_box
        cp_detects.append([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])

    for tracker in trackers:
        bbox = tracker.crowd.bounding_box
        cp_tracks.append([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])

    cp_detects = np.array(cp_detects)
    cp_tracks = np.array(cp_tracks)
    cost_matrix = np.zeros((len(cp_detects), len(cp_tracks)))

    for row, _ in enumerate(cost_matrix):
        cost_matrix[row, :] = np.sqrt(np.power(cp_detects[row, 0] - cp_tracks[:, 0], 2) +
                                      np.power(cp_detects[row, 1] - cp_tracks[:, 1], 2))
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    return cost_matrix


def _cosine_distance(row_array, col_array, data_is_normalized=False):
    if not data_is_normalized:
        row_array = np.asarray(row_array) / np.linalg.norm(row_array, axis=1, keepdims=True)
        col_array = np.asarray(col_array) / np.linalg.norm(col_array, axis=1, keepdims=True)
    return 1. - np.dot(row_array, col_array.T)


def _nn_cosine_distance(row_array, col_array):
    distances = _cosine_distance(row_array, col_array)
    return distances.min(axis=0)


def get_feature_cost_matrix(crowds: List[Crowd], trackers: List[KalmanBoxTracker]) -> np.ndarray:
    cost_matrix = np.zeros((len(crowds), len(trackers)))
    features_detections = np.array([[crowd.feature_vector] for crowd in crowds])
    features_track = np.array([tracker.crowd.feature_vector for tracker in trackers])

    for idx, features_detection in enumerate(features_detections):
        cost_matrix[idx, :] = _nn_cosine_distance(row_array=features_detection, col_array=features_track)
    # cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    return cost_matrix


def associate_detections_to_trackers(crowds: List[Crowd], trackers: List[KalmanBoxTracker], max_distance=0.9):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(crowds)), np.empty((0, 5), dtype=int)
    if len(crowds) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(crowds)), np.arange(len(trackers), dtype=int)

    cost_matrix = (1.0/100) * get_euclid_distance_cost_matrix(crowds=crowds, trackers=trackers, max_distance=100)
    # cost_matrix += 0.5 * get_feature_cost_matrix(persons_info=persons_info, trackers=trackers)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_indices, col_indices = linear_assignment(cost_matrix)
    unmatched_detections, unmatched_tracks, matches = [], [], []

    for col, track in enumerate(trackers):
        if col not in col_indices:
            unmatched_tracks.append(col)

    for row, crowd in enumerate(crowds):
        if row not in row_indices:
            unmatched_detections.append(row)

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(col)
            unmatched_detections.append(row)
        else:
            matches.append((row, col))
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_tracks)
