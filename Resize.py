import cv2

class Reisze(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]

        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

        landmarks = landmarks * [self.output_size / w, self.output_size / h]

        return {'image':image, 'landmarks':landmarks}