#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os

import chainer
import time
import numpy as np
import matplotlib.pyplot as plt
from chainer.datasets import tuple_dataset

import net

def all_filled(arr):
    for v in arr:
        if not v:
           return False
    return True

class MakeRandomMNIST:
    def __init__(self):
        self.train, self.test = chainer.datasets.get_mnist()
        self.xs, self.ts = self.train._datasets
           
    def get_random_dataset_nolabel(self, n):
        ret = np.zeros((n, 28*28), dtype=np.float32)
        for i in range(n):
            im = self.get_random_img_from_pos(np.random.rand())
            ret[i, :] = np.reshape(im, 28*28)
        return ret

    def get_random_dataset_with_label(self, n):
        images = np.zeros((n, 28*28), dtype=np.float32)
        labels = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            pos = np.random.rand()
            labels[i, :] = pos
            im = self.get_random_img_from_pos(pos)
            images[i, :] = np.reshape(im, 28*28)
        return chainer.datasets.TupleDataset(images, labels)

    def get_random_dataset_with_hot_vector(self, n):
        ndim = 100
        images = np.zeros((n, 28*28), dtype=np.float32)
        labels = np.zeros((n, ndim), dtype=np.float32)
        for i in range(n):
            pos = np.random.rand()
            lvec = np.eye(ndim, dtype=np.float32)[int(pos * ndim)]
            #print('***%f***\n'%pos)
            #print(lvec)
            labels[i, :] = lvec
            im = self.get_random_img_from_pos(pos)
            images[i, :] = np.reshape(im, 28*28)
        return chainer.datasets.TupleDataset(images, labels)

    def get_random_img_from_pos(self, pos):
        im = self.make_random_seq()
        assert(0 <= pos <= 1)
        x = int(pos * 9 * 28)
        return im[:, x:(x+28)]

    def make_random_seq(self):
        ts = self.ts
        xs = self.xs
        n = len(ts)
        filled = [False] * 10
        ret = np.zeros((28, 28 * 10), dtype=np.float32)
        while not all_filled(filled):
            ind = int(n * np.random.rand())
            num = ts[ind]
            img = np.reshape(xs[ind], (28, 28))
            ret[:, 28*num:28*(num+1)] = img
            filled[num] = True
        #plt.imshow(ret, cmap='gray')
        #plt.show()
        return ret

if __name__ == '__main__':
    t0 = time.time()
    print('start\n')
    m = MakeRandomMNIST()
    X = m.get_random_dataset_nolabel()
    for i in range(1000):
        #s = m.make_random_seq()
        im = m.get_random_img_from_pos(np.random.rand())
        #plt.imshow(im, cmap='gray')
        #plt.show()
    print('done: %f'%(time.time()-t0))
