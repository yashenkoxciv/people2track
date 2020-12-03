import os
import cv2 as cv
import numpy as np
import face_recognition


class Trackable:
    def __init__(self, identifier, features=None, bounding_boxes=None):
        self.identifier = identifier
        self.bounding_boxes = [] if not bounding_boxes else bounding_boxes
        self.features = [] if not features else features
        self.disappeared = 0

    def extend(self, feature, bounding_box):
        self.features.append(feature)
        self.bounding_boxes.append(bounding_box)
        self.disappeared = 0

    def get_centroid(self):
        return np.mean(self.features, 0).reshape(-1, 128)

    def __str__(self):
        return '<Trackable {0:03d} contains {1} examples>'.format(self.identifier, len(self.features))

    def __repr__(self):
        return str(self)


class CentroidTracker:
    def __init__(self, similarity_threshold, max_disappeared):
        self.next_id = 0
        self.trackables = {}
        self.left = []
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared

    def handle_disappeared(self):
        """
        Increase disappearance counter and delete tracks which does not have names
        :return: None
        """
        for trackable_id in list(self.trackables.keys()):
            trackable = self.trackables[trackable_id]
            trackable.disappeared += 1
            if trackable.disappeared > self.max_disappeared:
                del self.trackables[trackable_id]

    def update(self, image):
        """
        Update tracking system state
        :param image: current frame from video or stream
        :return: current_trackables (dict), detected objects
        """
        # obtain face locations in an image
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0)
        # obtain feature vectors for every face in an image
        face_features = face_recognition.face_encodings(image, face_locations)
        current_trackables = {}
        for bounding_box, feature in zip(face_locations, face_features):
            matched_trackable_id, matched_distance, matched_centroid = self.find_match(feature)
            if matched_trackable_id is None:  # create new trackable
                new_trackable_id = self.push_in(feature, bounding_box)
                current_trackables[new_trackable_id] = self.trackables[new_trackable_id]
            else:  # continue existing trackable
                trackable_to_extend = self.trackables[matched_trackable_id]
                trackable_to_extend.extend(feature, bounding_box)
                current_trackables[matched_trackable_id] = self.trackables[matched_trackable_id]
        # delete disappeared objects
        self.handle_disappeared()
        return current_trackables

    def push_in(self, feature, bounding_box):
        """
        Start new track.
        :param bounding_box: ndarray, (1, 4), array that enumerates (top, right, bottom, left) coordinates
        :param feature: feature vectors of the bounding box
        :return: identifier (int) of already started track
        """
        identifier = self.next_id
        self.next_id += 1
        trackable = Trackable(identifier, features=[feature], bounding_boxes=[bounding_box])
        self.trackables[identifier] = trackable
        return identifier

    def find_match(self, feature):
        """
        Looks for nearest neighbor in .trackable dict.
        :param feature: ndarray, (1, 128) feature vector to match
        :return: trackable_id (int), distance (float), centroid (ndarray, 128)
        """
        best_trackable_id = None
        best_distance = float('inf')
        best_centroid = None
        for trackable_id in self.trackables:
            trackable = self.trackables[trackable_id]
            centroid = trackable.get_centroid()
            distance = face_recognition.face_distance(centroid, feature)[0]
            if (distance < best_distance) and (distance < self.similarity_threshold):
                best_distance = distance
                best_centroid = centroid[0]
                best_trackable_id = trackable_id
        return best_trackable_id, best_distance, best_centroid

    def __str__(self):
        return f'<CentroidTracker size={len(self.trackables)}>'

    def __repr__(self):
        return str(self)
