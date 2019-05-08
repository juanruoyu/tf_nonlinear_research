#!/usr/bin/env python
import os
import cv2
import numpy as np
from scipy import io as scio
from config import config
from brainpp.oss import OSSPath
from pathlib import Path
import nori2 as nori

class Dataset():
    def __init__(self, dataset_name):
        'init the config and some hyperparameter for network'
        self.dataset_path = OSSPath('s3://awesome-neupeak/dataset/MNIST')
        self.minibatch_size = config.minibatch_size
        self.dataset_name = self.ds_name = dataset_name
        nori_files = {
            'train': self.dataset_path / 'train.nori',
            'validation': self.dataset_path / 'val.nori',
            'test': self.dataset_path / 'test.nori',
        }
        nori_file = nori_files[self.dataset_name]
        self.nr = nori.open(str(nori_file), 'r')

        count = 0
        for nid, _, _ in self.nr.scan(scan_data=False, scan_meta=False):
            count += 1
        self.instance_per_epoch = count

    def load(self):
        """The MNIST dataset is small, read them all into memory"""
        self.imgs = []
        self.labels = []

        for _, png, meta in self.nr.scan():
            img = cv2.imdecode(np.fromstring(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            label = meta['extra']['label']
            self.imgs.append(img)
            self.labels.append(label)
        self.instances = len(self.imgs)
        return self

    @property
    def instances_per_epoch(self):
        return self.instances

    @property
    def minibatchs_per_epoch(self):
        return self.instances // config.minibatch_size

    def instance_generator(self):
        for i in range(self.instances):
            img, label = self.imgs[i], self.labels[i]
            img = cv2.resize(img, config.image_shape)
            yield img.astype(np.float32)[:, :, np.newaxis], np.array(label, dtype=np.int32)


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
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5*img.shape[0], 5*img.shape[1], 1))
        cv2.imshow('', imggrid.astype('uint8'))
        c = chr(cv2.waitKey(0) & 0xff)
        if c == 'q':
            exit()
        imggrid = []

