import argparse
import os

import chainer
import time
import numpy as np
import matplotlib.pyplot as plt
from chainer.datasets import tuple_dataset
from PIL import Image, ImageDraw

center_point_list = np.empty((1, 2))
for row in range(2):
    for col in range(5):
        c_point = (14+(28*col), 14*(row+1)+(row*14))
        center_point_list = np.vstack((center_point_list, c_point))
center_point_list = np.delete(center_point_list,0,0)

class MakeRandomSelfdata:
    def __init__(self, img):
        self.img = Image.open(img).convert("L")
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
    
    def random_crop_in_area(self, left, upper, right, lower, label):
        image_list = np.empty(28*28,dtype=np.float32)
        randx = np.random.randint(left, right)
        randy = np.random.randint(upper, lower)
        crop_img = self.cropImage(randx, randy, 28, 28)
        c_point = [[float(randx)/(28*5), float(randy)/(28*2)]]
        c_point = np.array(c_point, dtype=np.float32)
        return crop_img, label, c_point
    def cropImage(self, center_x, center_y, height, width):
        left = center_x - width/2
        upper = center_y - height/2
        right = center_x + width/2
        lower = center_y + height/2
        c_img = self.img.crop((left, upper, right, lower))
        carr = np.asarray(c_img)
        carr = carr.flatten()
        carr = np.asarray(carr).astype(np.float32)
        return np.array(carr, dtype=np.float32)/255
    def convert_axis(self, label, movement):
        m = movement
        p = center_point_list
        crop_area = []
        if label == 0:
            crop_area = (p[label][0], p[label][1], p[label][0]+m, p[label][1]+m)
        elif label == 1:
            crop_area = (p[label][0]-m, p[label][1], p[label][0]+m, p[label][1]+m)
        elif label == 2:
            crop_area = (p[label][0]-m, p[label][1], p[label][0]+m, p[label][1]+m)
        elif label == 3:
            crop_area =  (p[label][0]-m, p[label][1], p[label][0]+m, p[label][1]+m)
        elif label == 4:
            crop_area = (p[label][0]-m, p[label][1], p[label][0], p[label][1]+m)
        elif label == 5:
            crop_area =(p[label][0], p[label][1]-m, p[label][0]+m, p[label][1])
        elif label == 6:
            crop_area = (p[label][0]-m, p[label][1]-m, p[label][0]+m, p[label][1])
        elif label == 7:
            crop_area = (p[label][0]-m, p[label][1]-m, p[label][0]+m, p[label][1])
        elif label == 8:
            crop_area = (p[label][0]-m, p[label][1]-m, p[label][0]+m, p[label][1])
        else:
            crop_area = (int(p[label][0]-m), int(p[label][1]-m), int(p[label][0]), int(p[label][1]))
        return crop_area
    def get_random_dataset_with_label(self, n):
        images = np.zeros((n, 28*28), dtype=np.float32)
        labels = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            '''
            pos = np.random.rand()
            labels[i, :] = pos
            im = self.get_random_img_from_pos(pos)
            images[i, :] = np.reshape(im, 28*28)
            '''
            label_0to9 = np.random.randint(10)
            labels[i, :] = label_0to9
            left, upper, right, lower = self.convert_axis(label_0to9, 5)
            img, _, _ = self.random_crop_in_area(left, upper, right, lower,label_0to9)
            images[i, :] = np.reshape(img, 28*28)
        return chainer.datasets.TupleDataset(images, labels)