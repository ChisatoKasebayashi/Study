import argparse
import os

import chainer
import time
import numpy as np
import matplotlib.pyplot as plt
from chainer.datasets import tuple_dataset
from PIL import Image, ImageDraw
import cv2

center_point_list = np.empty((1, 2))
for row in range(2):
    for col in range(5):
        c_point = (14+(28*col), 14*(row+1)+(row*14))
        center_point_list = np.vstack((center_point_list, c_point))
center_point_list = np.delete(center_point_list,0,0)

class MakeRandomSelfdata:
    def __init__(self, img):
        self.onehot_ratio = 1
        self.img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        h,w = self.img.shape[:2]
        self.onehot_w = int((w-28)/self.onehot_ratio)+1
        self.onehot_h = int((h-28)/self.onehot_ratio)+1
        self.rotation_angle = 180 #degree
    def random_crop_in_area(self, left, upper, right, lower, label):
        image_list = np.empty(28*28,dtype=np.float32)
        randx = np.random.randint(left, right)
        randy = np.random.randint(upper, lower)
        crop_img = self.cropImage(randx, randy, 28, 28)
        c_point = [[float(randx)/(28*5), float(randy)/(28*2)]]
        c_point = np.array(c_point, dtype=np.float32)
        return crop_img, label, c_point
    def cropImage(self, center_x, center_y, height, width):
        half_w = int(width/2)
        half_h = int(height/2)
        left = center_x - half_w
        upper = center_y - half_h
        right = center_x + half_w
        lower = center_y + half_h
        #c_img = self.img.crop((left, upper, right, lower))
        c_img = self.img[upper:lower, left:right]
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
    def get_random_img_from_pos(self, pos):
        '''
        im = self.make_random_seq()
        assert(0 <= pos <= 1)
        x = int(pos * 9 * 28)
        return im[:, x:(x+28)]
        '''
        x = int(pos * 9 * 28)
        im = self.cropImage(x, 14, 28, 28)
        return im
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
    def get_random_dataset_with_label(self, n):
        images = np.zeros((n, 28*28), dtype=np.float32)
        labels = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            label_0to9 = np.random.randint(10)
            labels[i, :] = label_0to9
            left, upper, right, lower = self.convert_axis(label_0to9, 5)
            img, _, _ = self.random_crop_in_area(left, upper, right, lower,label_0to9)
            images[i, :] = np.reshape(img, 28*28)
        return chainer.datasets.TupleDataset(images, labels)
    def get_random_dataset_with_x_coordinate(self, n):
        ndim = 1
        images = np.zeros((n, 28*28), dtype=np.float32)
        labels = np.zeros((n, ndim), dtype=np.float32)
        for i in range(n):
            pos = np.random.rand()
            lvec = pos
            #print('***%f***\n'%pos)
            #print(lvec)
            labels[i, :] = lvec
            im = self.get_random_img_from_pos(pos)
            images[i, :] = np.reshape(im, 28*28)
        return chainer.datasets.TupleDataset(images, labels)
    def get_random_dataset_with_hot_vector_2d(self, n):
        ndim_row = 20
        ndim_col = 50
        images = np.zeros((n, 28*28), dtype=np.float32)
        labels = np.zeros((n, ndim_row+ndim_col), dtype=np.float32)
        labels_row = np.zeros((n, ndim_row), dtype=np.float32)
        labels_col = np.zeros((n, ndim_col), dtype=np.float32)
        for i in range(n):
            posx = np.random.rand()
            posy = np.random.rand()
            lvec_r = np.eye(ndim_col, dtype=np.float32)[int(posx * ndim_col)]
            lvec_c = np.eye(ndim_row, dtype=np.float32)[int(posy * ndim_row)]
            lvec = np.append(lvec_r, lvec_c)
            labels[i, :] = lvec
            #print('(x,y) = ({0}, {1})'.format(int((posx*4*28)+14),int((posy*1*28)+14)))
            im = self.cropImage(int((posx*4*28)+14), int((posy*1*28)+14), 28, 28)
            images[i, :] = np.reshape(im, 28*28)
        return chainer.datasets.TupleDataset(images, labels)
    def get_random_dataset_with_one_hot_vector_2d(self, n):
        labels = np.zeros((n, self.onehot_w*self.onehot_h), dtype=np.float32)
        #print('get_random_dataset_with_one_hot_vector' + str(labels.shape))
        #print(onehot_w)
        #print(onehot_h)
        #print(onehot_w*onehot_h)
        images = np.zeros((n, 28*28), dtype=np.float32)
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            labels[i, :] = self.getLabel(posx,posy)
            images[i, :] = self.getImage(posx,posy)
        return chainer.datasets.TupleDataset(images, labels)
    def get_random_dataset_with_gentle_hot_vector_2d(self, n):
        devariation = 10
        random_nun = 20
        labels = np.zeros((n, self.onehot_w*self.onehot_h), dtype=np.float32)
        images = np.zeros((n, 28*28), dtype=np.float32)
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            images[i, :] = self.getImage(posx,posy)
            hotvec = self.getLabel(posx,posy)
            g_hotvec =  self.make_gentle_onehot_vec(hotvec)
            labels[i, :] = g_hotvec
        return chainer.datasets.TupleDataset(images, labels)
    
    def get_random_dataset_for_rcvae(self, n): # one dimentional gauss
        deviation = 15
        random_nun = 150
        labels = np.zeros((n, self.onehot_w*self.onehot_h), dtype=np.float32)
        images = np.zeros((n, 28*28), dtype=np.float32)
        context = np.zeros((n, self.onehot_w*self.onehot_h + (28*28)), dtype=np.float32)
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            images[i, :] = self.getImage(posx,posy) # imageをn個images[]にためてる
            hotvec = self.getLabel(posx,posy)
            g_hotvec =  self.make_gentle_onehot_vec(hotvec)
            labels[i, :] = g_hotvec
        return chainer.datasets.TupleDataset(labels, images)

    def make_gentle_onehot_vec(self, hotvec): # one dimentional gauss
        g_hotvec = hotvec.copy()
        deviation = 15
        random_nun = 200
        hotvec_l = hotvec.tolist()
        average = hotvec_l.index(1)
        g_hotvec[average] = 0
        rand_n = np.random.normal(average, deviation, random_nun)
        for n in range(random_nun):
            index = rand_n[n].astype('int32')
            if(index<0 or len(hotvec_l)<=index):
                continue
            g_hotvec[index] = g_hotvec[index] + 1
        ret = g_hotvec/max(g_hotvec)
        return ret
    
    def get_random_dataset_for_rcvae_with_2d_onehot(self, n): # two dimentional gauss
        deviation = 15
        random_nun = 150
        labels = np.zeros((n, self.onehot_w*self.onehot_h), dtype=np.float32)
        images = np.zeros((n, 28*28), dtype=np.float32)
        context = np.zeros((n, self.onehot_w*self.onehot_h + (28*28)), dtype=np.float32)
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            images[i, :] = self.getImage(posx,posy) # imageをn個images[]にためてる
            hotvec = self.getLabel(posx,posy)
            hotvec_l = hotvec.tolist()
            average = hotvec_l.index(1)
            g_hotvec =  self.make_gentle_onehot_vec_2d(np.reshape(hotvec,(self.onehot_h,self.onehot_w)))
            labels[i, :] = g_hotvec
        return chainer.datasets.TupleDataset(labels, images)
    
    def get_random_dataset_for_rcvae_with_2d_onehot_and_sincos(self, n): # two dimentional gauss
        deviation = 15
        random_nun = 150
        labels = np.zeros((n, self.onehot_w*self.onehot_h + 2), dtype=np.float32)
        images = np.zeros((n, 28*28), dtype=np.float32)
        context = np.zeros((n, self.onehot_w*self.onehot_h + (28*28)), dtype=np.float32)
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            #print(posx, posy, 'x, y')
            deg = np.random.randint(self.rotation_angle)
            #print(deg, 'deg in func')
            im, rad = self.getRotateImageAndRad(posx, posy, deg)
            images[i, :] = im
            hotvec = self.getLabel(posx,posy)
            hotvec_l = hotvec.tolist()
            average = hotvec_l.index(1)
            g_hotvec =  self.make_gentle_onehot_vec_2d(np.reshape(hotvec,(self.onehot_h,self.onehot_w)))
            angle = [np.sin(rad), (np.cos(rad)+1)/2]
            labels[i, :] = np.concatenate([g_hotvec, angle])
        return chainer.datasets.TupleDataset(labels, images)
    def get_random_dataset_for_rcvae_with_2d_GentleOnehotPosMap_and_1d_GentleOnehotTheta(self, n):
        angle_dim = self.rotation_angle*17
        labels = np.zeros((n, self.onehot_w*self.onehot_h + angle_dim), dtype=np.float32)
        images = np.zeros((n, 28*28), dtype=np.float32)
        context = np.zeros((n, self.onehot_w*self.onehot_h + (28*28)), dtype=np.float32)
        debug_data = np.zeros((n, 3), dtype=np.float32)
        
        print(angle_dim, 'angle_dim')
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            deg = np.random.randint(self.rotation_angle)
            #########image[condition]###########
            im, rad = self.getRotateImageAndRad(posx, posy, deg)
            images[i, :] = im 
            #########image[condition]###########
            
            #*********probMap[input]***********#
            hotvec = self.getLabel(posx,posy)
            hotvec_l = hotvec.tolist()
            average = hotvec_l.index(1)
            g_hotvec =  self.make_gentle_onehot_vec_2d(np.reshape(hotvec,(self.onehot_h,self.onehot_w)))
            #*********probMap[input]***********#
            angle_onehot = self.getLabel_specified_1d(deg, self.rotation_angle)
            angle_ghot = self.make_gentle_onehot_vec(angle_onehot)
            angle_ghot = np.tile(angle_ghot,17)
            #print(angle_ghot.shape, 'angle_ghot shape')
            #print(np.concatenate([g_hotvec, angle_ghot]).shape, 'cnca')
            #print(g_hotvec.shape, 'ghot_vec')
            labels[i, :] = np.concatenate([g_hotvec, angle_ghot])
            debug_data[i, :] = [posx, posy, deg]
        return chainer.datasets.TupleDataset(labels, images), debug_data
    
    def get_random_dataset_for_rcvae_with_2d_GentleOnehotPosMap_and_2d_GentleOnehotSinCos(self, n):
        labels = np.zeros((n, self.onehot_w*self.onehot_h + 40*80), dtype=np.float32) #posmap + angleMap　が入る
        images = np.zeros((n, 28*28), dtype=np.float32)
        angle = np.zeros((n, self.rotation_angle*2))
        debug_data = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            posx = int((np.random.rand()*(4*28+1))+14)
            posy = int((np.random.rand()*(1*28+1))+14)
            deg = np.random.randint(self.rotation_angle)
            #########image[condition]###########
            im, rad = self.getRotateImageAndRad(posx, posy, deg)
            images[i, :] = im 
            #########image[condition]###########
            
            #*********probPosMap[input]***********#
            hotvec = self.getLabel(posx,posy)
            hotvec_l = hotvec.tolist()
            average = hotvec_l.index(1)
            g_hotvec =  self.make_gentle_onehot_vec_2d(np.reshape(hotvec,(self.onehot_h,self.onehot_w)))
            #*********probPosMap[input]***********#
            
            #*********probAngleMap[input]***********#
            angle_ghot = self.make_angle_map(deg)
            #*********probAngleMap[input]***********#
            labels[i, :] = np.concatenate([g_hotvec, angle_ghot])
            debug_data[i, :] = [posx, posy, deg]
        return chainer.datasets.TupleDataset(labels, images), debug_data
    
    def make_angle_map(self, deg):
        w=80
        h=40
        im = np.zeros((h,w), dtype=np.float32)
        x = int(39*np.cos(np.deg2rad(deg))+40)
        y = int(39*np.sin(np.deg2rad(deg)))
        im[y][x] = 1
        ret = self.make_gentle_onehot_vec_2d_custom(im,w,h)
        return ret
    
    def make_gentle_onehot_vec_2d_custom(self, hotvec,w,h): # two dimentional gauss
        #print(hotvec.shape, 'hotvec shape')
        random_nun = 200
        deviation = [[5, 0], [0, 5]]
        #values = np.random.multivariate_normal(mu, sigma, 1000)
        g_hotvec = hotvec.copy()
        average_np = np.where(hotvec==1)
        #print(average_np, 'average')
        average = [average_np[1][0],average_np[0][0]]
        #print(average)
        g_hotvec[average[1],average[0]] = 0
        rand_n = np.random.multivariate_normal(average, deviation, random_nun)
        #print(rand_n.shape, 'rand_n')
        for n in range(len(rand_n)):
            #print(rand_n)
            index_x = rand_n[n][0].astype('int32')
            index_y = rand_n[n][1].astype('int32')
            if(index_x < 0 or w <= index_x):
                #print(index_x, 'index_x')
                continue
            if(index_y < 0 or h <= index_y):
                #print(index_y, 'index_y')
                continue
            #print('UnKO')
            g_hotvec[index_y][index_x] = g_hotvec[index_y][index_x] + 1
        g_hotvec = np.reshape(g_hotvec, h*w)
        #print(g_hotvec,'ghotvec')
        #print(max(g_hotvec),'max ')
        ret = g_hotvec/np.max(g_hotvec)
        #ret = g_hotvec
        #print(max(ret))
        #print(ret, 'ret')
        return ret
    
    def make_gentle_onehot_vec_2d(self, hotvec): # two dimentional gauss
        #print(hotvec.shape, 'hotvec shape')
        random_nun = 200
        deviation = [[5, 0], [0, 5]]
        #values = np.random.multivariate_normal(mu, sigma, 1000)
        g_hotvec = hotvec.copy()
        average_np = np.where(hotvec==1)
        #print(average_np, 'average')
        average = [average_np[1][0],average_np[0][0]]
        #print(average)
        g_hotvec[average[1],average[0]] = 0
        rand_n = np.random.multivariate_normal(average, deviation, random_nun)
        #print(rand_n.shape, 'rand_n')
        for n in range(len(rand_n)):
            #print(rand_n)
            index_x = rand_n[n][0].astype('int32')
            index_y = rand_n[n][1].astype('int32')
            if(index_x < 0 or self.onehot_w <= index_x):
                #print(index_x, 'index_x')
                continue
            if(index_y < 0 or self.onehot_h <= index_y):
                #print(index_y, 'index_y')
                continue
            #print('UnKO')
            g_hotvec[index_y][index_x] = g_hotvec[index_y][index_x] + 1
        g_hotvec = np.reshape(g_hotvec, self.onehot_h*self.onehot_w)
        #print(g_hotvec,'ghotvec')
        #print(max(g_hotvec),'max ')
        ret = g_hotvec/np.max(g_hotvec)
        #ret = g_hotvec
        #print(max(ret))
        #print(ret, 'ret')
        return ret
    def flipImage_horizonal_Next2_vartical(self, arr, w, h):
        img = np.reshape(arr, (w,h))
        mirror_img = img[:, ::-1]
        f_img = mirror_img[::-1, :]
        return np.reshape(f_img, w*h)
    
    def rotateImage_and_cut(self, deg, center, size):
        rot_mat = cv2.getRotationMatrix2D(center, deg, 1.0)
        rot_mat[0][2] += -center[0]+size[0]/2 # -(元画像内での中心位置)+(切り抜きたいサイズの中心)
        rot_mat[1][2] += -center[1]+size[1]/2 # 同上
        res = cv2.warpAffine(self.img, rot_mat, size)
        carr = res.flatten()
        return np.array(carr, dtype=np.float32)/255
    
    def getLabel(self,posx,posy):
        l = np.zeros((self.onehot_h, self.onehot_w), dtype=np.float32)
        l[int((posy-14)/self.onehot_ratio)][int((posx-14)/self.onehot_ratio)] = 1
        label = np.ravel(l)
        #print('getLabel' + str(label.shape))
        #print(self.onehot_w)
        #print(self.onehot_h)
        return label

    def getLabel_specified_1d(self, pos, size):
        l = np.zeros(size, dtype=np.float32)
        l[int(pos)] = 1
        label = np.ravel(l)
        return label
        
    def getImage(self, posx,posy):
        im = self.cropImage(posx, posy, 28, 28)
        np.reshape(im, 28*28)
        return im
    
    def getRotateImageAndRad(self, posx, posy, deg):
        im = self.cropImage(posx, posy, 28,28)
        #self.dispImage(im)
        #print('------------')
        res = self.rotateImage_and_cut(deg, (posx,posy), (28,28) )
        rad = rad=deg*(np.pi/180)
        return res, rad
    
    def getOnehotSize(self):
        return np.array((self.onehot_h, self.onehot_w),np.int32)
    
    def dispImage(self,img_vec):
        #title = 'Label number is ('+ str(label_x) + ',' + str(label_y) + ')' 
        pixels = (img_vec * 256).reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.axis("off")
        #plt.title(title)
        plt.show()