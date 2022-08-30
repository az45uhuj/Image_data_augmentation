import torch
import numpy as np
import cv2

class RandomRotate(object):
    """Random rotate images with face landmarks"""
    def __init__(self):
        self.p = 0
        self.angle = 0

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        self.p = torch.rand(1)
        if p > 0.5:
            self.angle = np.random.uniform(-15, 15)

            # grab the dimensions of the image and determine the centre
            h, w = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            # grab the rotation matrix (apply the negative of the angle to rotate clockwise)
            # , then grab the sine and cosine (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), self.angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin)) + (w * cos)
            nH = int((h * cos)) + (w * sin)

            # adjust the rotation matrix take into account translation
            M[0, 2] += (nW / 2 ) - cX
            M[1, 2] += (nH / 2 ) - cY

            # perform the actual rotation and return the image
            image = np.array(image)
            image = cv2.wrapAffine(image, M, (nW, nH), borderMode=1)
            M2 = np.eye(3)
            M2[:2, :] = M

            a = np.ones(5).reshape(5, 1)
            temp1 = np.hstack((landmarks, a))
            tmp = np.dot(M2, temp1.transpose)
            landmarks = (tmp[:2, :] / tmp[-1, :]).transpose()

            return {'image':image, 'landmarks': landmarks}