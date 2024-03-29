#!/usr/bin/env python
import os
import cv2
import numpy as np
from scipy import io as scio
from config import config

class Dataset():
    def __init__(self, dataset_name):
        'init the config and some hyperparameter for network'
        self.dataset_path = config.dataset_path
        self.minibatch_size = config.minibatch_size
        self.ds_name = dataset_name
        if config.use_extra_data:
            train_meta = ([os.path.join(self.dataset_path, 'train_32x32.mat'),
                           os.path.join(self.dataset_path, 'extra_32x32.mat')], 604388)
        else:
            train_meta = ([os.path.join(self.dataset_path, 'train_32x32.mat')], 73257)

        dataset_meta = {
            'train': train_meta,
            'test': ([os.path.join(self.dataset_path, 'test_32x32.mat')], 26032),
        }
        self.files, self.instances = dataset_meta[dataset_name]

    def load(self):
        '''load your dataset and transform to mat for tensorflow format tensor'''
        datas_list, labels_list = [], []
        for f in self.files:
            samples = scio.loadmat(f)
            datas_list.append(samples['X'])
            labels_list.append(samples['y'])
        datas = np.concatenate(datas_list, axis=3)
        labels = np.concatenate(labels_list, axis=0)
        self.samples_mat = {
            'X': datas,
            'Y': labels,
        }
        return self

    @property
    def instances_per_epoch(self):
        return self.instances

    @property
    def minibatchs_per_epoch(self):
        return self.instances // config.minibatch_size

    def instance_generator(self):
        for i in range(self.instances):
            img = self.samples_mat['X'][:, :, :, i]
            label = self.samples_mat['Y'][i, :][0]
            if label == 10:
                label = 0
            img = cv2.resize(img, config.image_shape)
            yield img.astype(np.float32), np.array(label, dtype=np.int32)


if __name__ == "__main__":
    ds = Dataset('train')
    ds = ds.load()
    gen = ds.instance_generator()

    imggrid = []
    while True:
        for i in range(25):
            img, label = next(gen)
            cv2.putText(img, str(label), (0, config.image_shape[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            imggrid.append(img)

        imggrid = np.array(imggrid).reshape((5, 5, img.shape[0], img.shape[1], img.shape[2]))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5*img.shape[0], 5*img.shape[1], 3))
        cv2.imshow('', imggrid.astype('uint8'))
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []

