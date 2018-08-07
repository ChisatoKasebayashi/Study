import pandas as pd
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tools.geometry

img = Image.open('hlfield.png')
#resize_img = img.resize((9,6))
nimg = np.array(img)
#target_path = '../data/LOG_G/log201807211402.csv'
#target_path = './log201807241933.csv'
target_path = sys.argv[1]
data = pd.DataFrame()
data = pd.read_csv(target_path,header=None)
#print(data)

l_ball = np.array(data.iloc[:,0:2])
l_pole0 = np.array(data.iloc[:,2:4])
l_pole1 = np.array(data.iloc[:,4:6])
g_ball = np.array(data.iloc[:,6:8])
g_pole0 = np.array(data.iloc[:,8:10])
g_pole1 = np.array(data.iloc[:,10:12])
g_robot = np.array(data.iloc[:,12:15])
print(l_ball)

fig, ax = plt.subplots()
fig.set_tight_layout(True)

print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))

def convertPoint(x, y, im_w, im_h):
    im_w = im_w-140
    im_h = im_h-140
    x = (x/9.0)*im_w+(im_w/2)+70
    y = (y/6.0)*im_h+(im_h/2)+70
    return x,y

def update(i):
    if i !=0:
        plt.cla()
    label = 'timestep {0}'.format(i)
    im_w = 1040
    im_h = 740
    ax.set_xlim(0,1040)
    ax.set_ylim(0,740)
    ax.imshow(nimg)
    step = i*10
    if step < len(l_ball):
#        lbx, lby =  convertPoint(robot[step][0]+l_ball[step][0],robot[step][1]+l_ball[step][1],im_w, im_h)
        #g_ball = tools.geometry.coord_trans_local_to_global(robot[step],l_ball[step])
        bx, by = convertPoint(g_ball[step][0],g_ball[step][1],im_w, im_h)
        rx, ry = convertPoint(g_robot[step][0],g_robot[step][1],im_w, im_h)
        p0x, p0y = convertPoint(g_pole0[step][0],g_pole0[step][1],im_w, im_h)
        p1x, p1y = convertPoint(g_pole1[step][0],g_pole1[step][1],im_w, im_h)

    else:
        step = len(l_ball)-1
#        lbx, lby =  convertPoint(robot[step][0]+l_ball[step][0],robot[step][1]+l_ball[step][1],im_w, im_h)
        bx, by = convertPoint(g_ball[step][0],g_ball[step][1],im_w, im_h)
        rx, ry = convertPoint(g_robot[step][0],g_robot[step][1],im_w, im_h)
        p0x, p0y = convertPoint(g_pole0[step][0],g_pole0[step][1],im_w, im_h)
        p1x, p1y = convertPoint(g_pole1[step][0],g_pole1[step][1],im_w, im_h)
    
    ax.scatter(bx, by,s=80,c='orange')
#    ax.scatter(lbx, lby,s=100, c='m')
    ax.scatter(rx, ry,s=120,facecolors='none',edgecolors='black')
    l = 25
    th = g_robot[step][2]
    lx = [rx, rx + l*np.cos(th)]
    ly = [ry, ry + l*np.sin(th)]
    ax.plot(lx,ly,'-', c='black')
    ax.scatter(p0x, p0y, s=200,c='gold')
    ax.scatter(p1x, p1y, s=200,c='gold')
#    print('x:'+str(ball[i][0])+'y:'+str(ball[i][1])+'\n')
    ax.set_xlabel(label)
    return ax

if __name__ == '__main__':
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(l_ball)/10), interval=1, repeat=False)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()
    sys.exit()
