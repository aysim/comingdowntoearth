from torchvision import transforms
import cv2
import torch
import random
import numpy as np

class RandomHorizontalFlip(object):

    def __call__(self, sample):
        if random.random() < 0.5:
            for elem in sample.keys():
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp
        return sample

class ToTensor(object):

    def __call__(self, sample):
        tanh_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        for elem in sample.keys():
            tmp = sample[elem]
            tmp = np.array(tmp, dtype=np.float32).transpose((2, 0, 1))
            tmp /= 255.0
            sample[elem] = torch.from_numpy(tmp)
            sample[elem] = tanh_norm(sample[elem])
        return sample
