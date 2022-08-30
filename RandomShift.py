import cv2
import torch
import numpy as np


class RandomShift(object):
    """Random shift image with face landmatks"""
    def __init__(self):
        self.p = 0
        self.shift = [0, 0]

    def __call__(self, sample):
        seed = int(torch.rand(1)*100)
        rng = np.random.default_rng(seed)
        image, landmarks = sample['image'], sample['landmarks']
        h,w = image.shape[:2]

        # here only shift 10% of weight and 10% of height
        w_shift = w * 0.1 if w * 0.1 > 1 else 1
        h_shift = h * 0.1 if h * 0.1 > 1 else 1
        self.shift[0] = rng.integers(low=-w_shift, high= w_shift, size=1)
        self.shift[1] = rng.integers(low=-h_shift, high= h_shift, size=1)

        # thanslatin matrix
        M = np.float32([[1, 0, self.shift[0]], [0, 1, self.shift[1]]])

        (rows, cols) = h, w
        image = np.array(image)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        M2 = np.eye(3).astype('float')
        M2[:2,:] = M

        # five landmarks
        a = np.ones(5).reshape(5, 1).astype('float')
        tmp1 = np.hstack((landmarks, a))
        tmp = np.dot(M2, tmp1.transpose())
        landmarks = (tmp[:2, :] / tmp[-1, :]).transpose()

        return {'image':image, 'landmarks':landmarks}


