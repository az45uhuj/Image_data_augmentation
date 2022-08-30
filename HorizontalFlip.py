import torch
import cv2
import numpy as np

class RandomHorizontalFlip():
    """Random horizontal flip image and face landmarks"""
    def __init__(self):
        self.p = 0

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmark']
        h, w = image.shape[:2]

        self.p = torch.rand(1)
        ldmks = landmarks.copy()

        if self.p > 0.5:
            img = cv2.flip(image, 1)
            landmarks[:, 0] = np.abs(w - landmarks[:, 0])

            # Here there are only five face landmarks (left_eye_left_corner, right_eye_right_corner, nose_tip,
            # mouth_left_corner, mouth_right_corner). Because after flipping image, the order of landmarks changed,
            # we need to reorder the landmarks.
            ldmks = landmarks[[1, 0, 2, 4, 3], :]
        else:
            img = image

        return {'image':img, 'landmarks':ldmks}
