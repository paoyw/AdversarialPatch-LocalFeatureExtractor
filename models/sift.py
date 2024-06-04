import cv2
import numpy as np
from torch import nn


class SIFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = cv2.SIFT_create()

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Please use to_cv2')

    def to_cv2(self, img):
        img = (255 * img.squeeze(dim=0)).cpu().numpy()
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        kp, desc = self.extractor.detectAndCompute(img, None)
        return kp, desc
