import torch
import cv2
import numpy as np

class RandomCrop(object):
    """Radom crop image with face landmarks"""
    def __init__(self):
        self.p = 0

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]

        # face bounding box
        maxy, miny = int(max(landmarks[:, 1])), int(min(landmarks[:, 1]))
        maxx, minx = int(max(landmarks[:, 0])), int(min(landmarks[:, 0]))

        self.p = torch.rand(1)
        ldmks = landmarks.copy()
        if p > 0.5:
            # (0.2, 0.9): how much you want to crop between the bbox and image edge.
            p0 = np.random.uniform(0.2, 0.9, 4)
            dis_minx, dis_maxx, dis_miny, dis_maxy = minx * p[0], maxx * p[1], miny * p[2], maxy * p[3]
            ix, iy = int(minx - dis_minx), int(miny - dis_miny)
            ax, ay = int(maxx + dis_maxx), int(maxy + dis_maxy)
            lx = int(ax - ix)
            ly = int(ay - iy)

            img = image[iy:iy + ly, ix:ix+lx]
            ldmks = lanmarks - [ix ,iy]
        else:
            img = image

            return {'image': img, 'landmarks':ldmks}

        

