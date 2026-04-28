try:
    import manconfig as cfg
    from utils import brush_mask, get_brush_deltas, timing, PlanetConfig, random_masking
except:
    from . import manconfig as cfg
    from .utils import brush_mask, get_brush_deltas, timing, PlanetConfig, random_masking
'''
#############################################
MOSTLY DEPRECATED
TODO Move the functions that are still to sketch_gen.py
#############################################
'''

import math
import cv2 
from PIL import Image as img
from PIL.Image import Image
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import os
import random
import json
from scipy import ndimage
import warnings

def edges(img,t1,t2):
    return cv2.Canny(img,t1,t2,None,5)
def checksurroundings(img,h,w,size):
    dirs = [False]*4
    for i in range(1,size+1):
        if(img[h+i][w] != 0):
            dirs[0] = True
        if(img[h][w+i] != 0):
            dirs[1] = True
        if(img[h-i][w] != 0):
            dirs[2] = True
        if(img[h][w-i] != 0):
            dirs[3] = True
        if dirs.count(False)==0:
            return True
    return dirs.count(False)==0
        

def getvals(img):
    # res = []
    # height = len(img)
    # width = len(img[0])
    # with tqdm(total=height*width) as bar:
    #     for h in range(height):
    #         width = len(img[h])
    #         for w in range(width):
    #             bar.update(1)
    #             px = img[h][w]
    #             if not px in res:
    #                 res.append(px)
    # res.sort()
    # return res
    return np.unique(img)
            

def fill(img,size):
    height = len(img)
    width = len(img[0])
    #res = img
    with tqdm(total=height*width) as bar:
        for h in range(height):
            width = len(img[h])
            for w in range(width):
                bar.update(1)
                if(w<size or w+size>=width or h<size or h+size>=height):
                    continue
                if(checksurroundings(img,h,w,size)):
                    img[h][w] = 255
    res = img
    return res

def filter(img,t):
    height = len(img)
    width = len(img[0])
    with tqdm(total=height*width) as bar:
        for h in range(height):
            for w in range(width):
                bar.update(1)
                if img[h][w] > t:
                    img[h][w] = 255
                else:
                    img[h][w] = 0
    return img

def split(img):
    height = len(img)
    width = len(img[0])
    img0 = img[0:height,0:width//2]
    img1 = img[0:height,width//2:]
    return img0,img1

def tile(img):
    height = len(img)
    width = len(img[0])
    img0 = img[0:height//2,0:width//2]
    img1 = img[0:height//2,width//2:]
    img2 = img[height//2:,0:width//2]
    img3 = img[height//2:,width//2:]
    return [[img0, img1],
            [img2, img3]]

def spin(img,amnt):
    height = len(img)
    img0 = img[0:height,0:amnt]
    img1 = img[0:height,amnt:]
    return np.concatenate((img1,img0), axis = 1)

def checkneighbours(img,h,w,t=10):
    count = 0
    vals = [0,1,-1]
    height = len(img)
    width = len(img[0])
    for val1 in vals:
        for val2 in vals:
            if img[(h+val1)%height][(w+val2)%width] < t:
                count+=1
    return count>4

def removescraps(img,t=10):#needs bw image
    height = len(img)
    width = len(img[0])
    imgcopy = np.ndarray.copy(img)
    with tqdm(total=height*width) as bar:
        for h in range(height):
            for w in range(width):
                bar.update(1)
                if not checkneighbours(img,h,w,t): 
                    imgcopy[(h)%height][(w)%width] = 255
                else:
                    imgcopy[(h)%height][(w)%width] = 0
                
    return imgcopy

def getobjects(img,cols):
    height = len(img)
    width = len(img[0])
    res=[]
    for i in range(len(cols)):
        res.append([])
    #res = img
    with tqdm(total=height*width) as bar:
        for h in range(height):
            width = len(img[h])
            for w in range(width):
                bar.update(1)
                for i in range(len(cols)):
                    n = nearest(img[h][w],cols)
                    if n == cols[i]:
                        res[i].append([h,w])
    return res

def getcenter(a):
    sumX,sumY=0,0
    for pos in a:
        sumX+=pos[0]
        sumY+=pos[1]
    return sumX//len(a),sumY//len(a)

def removeoutliers(cX,cY,dist,a=[]):
    for pos in a:
        if (cX-pos[0])**2 + (cY-pos[1])**2 > dist:
            a.remove(pos)
    return a


def normalize(img,cols):
    height = len(img)
    width = len(img[0])
    #res = img
    with tqdm(total=height*width) as bar:
        for h in range(height):
            width = len(img[h])
            for w in range(width):
                bar.update(1)
                img[h][w] = nearest(img[h][w],cols)
    res = img
    return res

def nearest(col,cols):
    mn = 256
    mnc = 0
    for c in cols:
        if abs(c-col)<mn:
            mn = abs(c-col)
            mnc = c
    return mnc
def getXaxis(img,pos=[]):
    print('Getting X-axis')
    if len(pos)==0:
        height = len(img)
        width = len(img[0])
        wMax,wMin = 0,width
        with tqdm(total=height*width) as bar:
            for h in range(height):
                width = len(img[h])
                for w in range(width):
                    bar.update(1)
                    if img[h][w] == 255:
                        if w>wMax:
                            wMax = w
                        if w<wMin:
                            wMin = w
        return (wMax+wMin)//2
    else:
        width = len(img[0])
        wMax,wMin = 0,width
        with tqdm(total=len(pos)) as bar:
            for p in pos:
                h,w=p[0],p[1]
                bar.update(1)
                if w>wMax:
                    wMax = w
                if w<wMin:
                    wMin = w
        return (wMax+wMin)//2


def getYaxis(img,pos=[]):
    print('Getting Y-axis')
    if len(pos)==0:
        height = len(img)
        width = len(img[0])
        hMax,hMin = 0,height
        with tqdm(total=height*width) as bar:
            for h in range(height):
                for w in range(width):
                    bar.update(1)
                    if img[h][w] == 255:
                        if h>hMax:
                            hMax = h
                        if h<hMin:
                            hMin = h
        return (hMax+hMin)//2
    else:
        height = len(img)
        width = len(img[0])
        hMax,hMin = 0,height
        with tqdm(total=len(pos)) as bar:
            for p in pos:
                h,w=p[0],p[1]
                bar.update(1)
                if h>hMax:
                    hMax = h
                if h<hMin:
                    hMin = h
        return (hMax+hMin)//2


def getPos(mask):
    pos = []
    if mask.dtype == bool:
        return np.argwhere(mask)
    return np.argwhere(mask>50)

def numpy_translate(base: np.ndarray, amnt: list) -> np.ndarray:
    '''
    Translate the array by a given amount.
    Args:
        base (np.ndarray): The array to translate.
        amnt (list): The percentage amount to translate by [y, x] relative to the height of the array.
    Returns:
        np.ndarray: The translated array.
    '''
    H, W = base.shape[:2]
    translation = (np.array(amnt)*H).astype(int)
    to_return = np.roll(base, translation, axis=(0,1))
    return to_return

def numpy_reflect(base: np.ndarray, x_line: int) -> np.ndarray:
    '''
    Reflects the array about a vertical line.

    Args:
        base (np.ndarray): The array to reflect.
    
    Returns:
        np.ndarray: The reflected array.
    '''

    reflected_base = np.roll(base, -x_line, axis=1)
    reflected_base = np.flip(reflected_base, axis=1)
    reflected_base = np.roll(reflected_base, x_line, axis=1)
    return reflected_base


def reflect(base,mask,axis,pos=[]):
    #print('Reflecting...')
    if len(pos)==0:
        oldmap = np.ndarray.copy(base)
        oldmask = np.ndarray.copy(mask)
        #obj = getobjects(mask,[255])[0]
        height = len(mask)
        width = len(mask[0])
        with tqdm(total=height*width) as bar:
            for h in range(height):
                for w in range(width):
                    bar.update(1)
                    if oldmask[h][w] > 50:
                        base[h][w] = np.ndarray.copy(oldmap[h][2*axis - w])
                        base[h][2*axis - w] = np.ndarray.copy(oldmap[h][w])

                        base[h][w+width] = np.ndarray.copy(oldmap[h][2*axis - w+width])
                        base[h][2*axis - w+width] = np.ndarray.copy(oldmap[h][w+width])

                        mask[h][w] = (oldmask[h][2*axis - w])
                        mask[h][2*axis - w] = (oldmask[h][w])
        return base,mask
    else:
        oldmap = np.ndarray.copy(base)
        oldmask = np.ndarray.copy(mask)
        width = len(mask[0])
        for p in pos:
            h,w=p[0],p[1]
            base[h][w] = np.ndarray.copy(oldmap[h][2*axis - w])
            base[h][2*axis - w] = np.ndarray.copy(oldmap[h][w])

            base[h][w+width] = np.ndarray.copy(oldmap[h][2*axis - w+width])
            base[h][2*axis - w+width] = np.ndarray.copy(oldmap[h][w+width])

            mask[h][w] = (oldmask[h][2*axis - w])
            mask[h][2*axis - w] = (oldmask[h][w])
            p[1] = 2*axis - w
        return base,mask,pos


def translate(base,mask,x,y):
    oldbase = np.ndarray.copy(base)
    oldmask = np.ndarray.copy(mask)
    height = len(mask)
    width = len(mask[0])
    with tqdm(total=height*width) as bar:
        for h in range(height):
            iter = range(width)
            if x < 0:
                iter = range(width)
            for w in iter:
                bar.update(1)
                if oldmask[h][w] > 30:
                    if w+x>=width:
                        x-=width
                    if w+x<0:
                        x+=width
                    if h+y>=height:
                        x-=height
                    if h+y<0:
                        x+=height
                    base[h][w] = [19,7,3]
                    base[h+y][w+x] = np.ndarray.copy(oldbase[h][w])
                    base[h][w+width] = [19,7,3]
                    base[h+y][w+x+width] = np.ndarray.copy(oldbase[h][w+width])
                    mask[h][w] = 0
                    mask[h+y][w+x] = (oldmask[h][w])
    return base,mask

def remove(base,mask,pos=[],colour=[19,7,3],alpha=1):
    if len(pos)==0:
        oldbase = np.ndarray.copy(base)
        oldmask = np.ndarray.copy(mask)
        height = len(mask)
        width = len(mask[0])
        with tqdm(total=height*width) as bar:
            for h in range(height):
                iter = range(width)
                for w in iter:
                    bar.update(1)
                    if oldmask[h][w] > 30:
                        base[h][w][:3] = colour
                        base[h][w+width][:3] = colour
                        if base[h][w].size > 3:
                            base[h][w][3] = alpha*255
                            base[h][w+width][3] = alpha*255

        return base,mask
    else:
        oldmask = np.ndarray.copy(mask)
        width = len(mask[0])
        for p in pos:
            h,w=p[0],p[1]
            if oldmask[h][w] > 30:
                base[h][w] = colour
                base[h][w+width] = colour
                if alpha < 1:
                    base[h][w][3] = alpha*255
                    base[h][w+width][3] = alpha*255
        return base,mask,pos



def compoundreflect(base,masks,axises,poss,selection = '1'):
    #for i in range(len(selection)-1,-1,-1):
    length = min(len(axises),len(selection))
    for i in range(length):
        if selection[length-i-1] == '1':
            base,masks[i],poss[i] = reflect(base,np.ndarray.copy(masks[i]),axises[i],poss[i])
    return base

def interp(base: np.ndarray, y: int, x:int) -> np.ndarray:
    '''
    Bilinear interpolation for a pixel on the base image.
    Does the interpolation in place.

    Args:
        base (np.ndarray): The base image.
        y (int): The y coordinates of the pixels.
        x (int): The x coordinates of the pixels.
    Returns:
        np.ndarray: The array of interpolated colours
    '''

    # From https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    y = np.asarray(y)
    x = np.asarray(x)

    x0 = np.round(x).astype(int) - 1
    x1 = x0 + 2
    y0 = np.round(y).astype(int) - 1
    y1 = y0 + 2

    x0 = np.clip(x0, 0, base.shape[1]-1)
    x1 = np.clip(x1, 0, base.shape[1]-1)
    y0 = np.clip(y0, 0, base.shape[0]-1)
    y1 = np.clip(y1, 0, base.shape[0]-1)

    Ia = base[ y0, x0 ]
    Ib = base[ y1, x0 ]
    Ic = base[ y0, x1 ]
    Id = base[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    res = ((Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T)//4
    return res


def cv2_rotate(mask: np.ndarray, angle: float, center: tuple) -> np.ndarray:
    '''
    Rotate a mask image by angle degrees around center using opencv affine rotation.
    Args:
        mask (np.ndarray): The mask image.
        angle (float): The angle to rotate by in degrees.
        center (tuple(cx: float, cy: float)): The center to rotate around.
    Returns:
        np.ndarray: The rotated mask image.
    '''
    h, w = mask.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_mask = cv2.warpAffine(mask.astype(np.float64), rot_mat, (w, h))

    if mask.dtype == bool or mask.dtype == np.bool_:
        new_mask = new_mask > 0.5
    else:
        new_mask = new_mask.astype(mask.dtype)
    return new_mask
    
def nearest_rotate(mask: np.ndarray, angle: float, center: tuple) -> np.ndarray:
    """
    Rotate the mask around it's center and use nearest neighbour interpolation.
    """
    h, w = mask.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_mask = cv2.warpAffine(mask.astype(np.float64), rot_mat, (w, h), flags=cv2.INTER_NEAREST)

    if mask.dtype == bool or mask.dtype == np.bool_:
        new_mask = new_mask > 0.5
    else:
        new_mask = new_mask.astype(mask.dtype)
    return new_mask

def numpy_rotate(base: np.ndarray, mask: np.ndarray, angle: float) -> tuple:
    '''
    Rotate the base and mask numpy arrays by angle degrees.

    Args:
        base (np.ndarray): The base image (either sketch or DEM).
        mask (np.ndarray): The mask image.
        angle (float): The angle to rotate by in degrees.
    
    Returns:
        tuple: The rotated base image, mask and new positions.
    '''
    warnings.warn('numpy_rotate is deprecated, use cv2_rotate instead', DeprecationWarning)
    assert base.shape[:2] == mask.shape[:2]
    pos = getPos(mask)
    cx = (np.max(pos[:,1]) + np.min(pos[:,1]))//2
    cy = (np.max(pos[:,0]) + np.min(pos[:,0]))//2
    angle = math.pi*angle/180
    s = math.sin(angle)
    c = math.cos(angle)
    h, w = mask.shape
    rot = lambda y, x: ((x-cx)*s+(y-cy)*c + cy, (x-cx)*c-(y-cy)*s + cx)
    new_pos = np.array([rot(p[0], p[1]) for p in pos])

    new_base = np.zeros_like(base)
    new_mask = np.zeros_like(mask)

    for i, (ry, rx) in enumerate(new_pos):
        if round(ry) < 0 or round(ry) >= new_base.shape[0]:
            continue
        y, x = pos[i]
        iy, ix = round(ry), round(rx)
        new_base[iy, ix%w] = base[y, x]
        new_mask[iy, ix%w] = mask[y, x]

    img.fromarray(new_base).save('rotated.png')

    bool_new_mask = new_mask>0
    b_mask = brush_mask(bool_new_mask, 1)
    # test = np.zeros_like(new_base)
    # test[b_mask] = [255, 255, 255, 255]
    # img.fromarray(test).save('b_mask_full.png')
    b_mask = np.logical_xor(b_mask, bool_new_mask)
    # test = np.zeros_like(new_base)
    # test[new_mask>0] = [255, 255, 255, 255]
    # img.fromarray(test).save('new_mask.png')
    # test = np.zeros_like(new_base)
    # test[b_mask] = [255, 255, 255, 255]
    # img.fromarray(test).save('b_mask.png')
    empty_pixels = np.where(b_mask)
    new_base[empty_pixels] = interp(new_base, empty_pixels[0], empty_pixels[1])
    # img.fromarray(new_base).save('interpolated.png')
    return new_base, new_mask, new_pos


def rotate(base,mask,angle,origin,pos=[]):
    warnings.warn('rotate is deprecated, use cv2_rotate instead', DeprecationWarning)
    #print('Rotating...')
    basecopy = np.ndarray.copy(base)
    maskcopy = np.ndarray.copy(mask)
    height = len(mask)
    width = len(mask[0])
    if len(pos)==0:

        new = []
        angle = math.pi*angle/180
        s = math.sin(angle)
        c = math.cos(angle)
        height = len(mask)
        width = len(mask[0])
        with tqdm(total=height*width) as bar:
            for h in range(height):
                for w in range(width):
                    bar.update(1)
                    if mask[h][w] > 20:
                        y, x = (h-origin[1]), (w-origin[0])
                        W = x*c-y*s
                        H = (x*s+y*c)
                        H+=origin[1]
                        W+=origin[0]
                        H,W = check(H,W,height,width)
                        base[h][w+width] = [19,7,3]
                        base[h][w] = [19,7,3]
                        mask[h][w] = 0
                        new.append([H,W,h,w])
    else:
        new = []
        angle = math.pi*angle/180
        s = math.sin(angle)
        c = math.cos(angle)
        height = len(mask)
        width = len(mask[0])
        for p in pos:
            h,w=p[0],p[1]
            y, x = (h-origin[1]), (w-origin[0])
            W = x*c-y*s
            H = (x*s+y*c)
            H+=origin[1]
            W+=origin[0]
            H,W = check(H,W,height,width)
            base[h][w+width][:3] = [0, 0, 0]
            base[h][w][:3]  = [0, 0, 0]
            if base[h][w].size > 3:
                base[h][w][3] = 0
                base[h][w+width][3] = 0
            mask[h][w] = 0
            new.append([H,W,h,w])

    for px in new:

        H,W = int(round(px[0])),int(round(px[1]))
        h,w = int((px[2])),int((px[3]))
        base[H][W] = np.ndarray.copy(basecopy[h][w])
        base[H][W+width] = np.ndarray.copy(basecopy[h][w+width])
        mask[H][W] = maskcopy[h][w]

    for px in new:

        H,W = int(round(px[0])),int(round(px[1]))
        h,w = int((px[2])),int((px[3]))
        for i in range(-1,2):
            for j in range(-1,2):
                y,x = (H+j) % height, (W+i) % width
                if mask[y][x] <= 50 or (base[y][x].size > 3 and base[y][x][3] < 50):
                    cols=[]
                    y,x = (H+j) % height, (W+i-1) % width
                    cols.append(base[y][x][:3])
                    y,x = (H+j) % height, (W+i) % width
                    cols.append(base[y][x][:3])
                    y,x = (H+j+1) % height, (W+i-1) % width
                    cols.append(base[y][x][:3])
                    y,x = (H+j+1) % height, (W+i) % width
                    cols.append(base[y][x][:3])
                    bi = bilinearInterp(px[0]+j,px[1]+i,cols) 
                    y,x = (H+j) % height, (W+i) % width
                    mask[y][x] = 255
                    base[y][x][:3] = np.ndarray.copy(bi)
                    if base[y][x].size > 3:
                        base[y][x][3] = 255
                    cols=[]
                    y,x = (H+j) % height, (W+i-1) % width
                    cols.append(base[y][x+width][:3])
                    y,x = (H+j) % height, (W+i) % width
                    cols.append(base[y][x+width][:3])
                    y,x = (H+j+1) % height, (W+i-1) % width
                    cols.append(base[y][x+width][:3])
                    y,x = (H+j+1) % height, (W+i) % width
                    cols.append(base[y][x+width][:3])
                    bi = cols[1] # bilinearInterp(px[0]+j,px[1]+i+width,cols) 
                    y,x = (H+j) % height, (W+i) % width
                    base[y][x+width][:3] = np.ndarray.copy(bi)
                    if base[y][x+width].size > 3:
                        base[y][x+width][3] = 255
                    base[y][x+width] = np.ndarray.copy(base[y-1][x+width])
                    # base[y][x+width] = [255, 255, 255, 255]
        base[H][W] = np.ndarray.copy(basecopy[h][w])
        base[H][W+width] = np.ndarray.copy(basecopy[h][w+width])
        mask[H][W] = maskcopy[h][w]



    return base,mask,pos
def check(y,x,height,width):
    if round(x)>=width:
        x-=width
    if round(x)<0:
        x+=width
    if round(y)>=height:
        y-=height
    if round(y)<0:
        y+=height
    return y,x

def bilinearInterp(x,y,cols=[0,0,0]*4):
    x-=int(x)
    y-=int(y)
    for i in range(3):
        if grayarr(cols[i])<20:
            cols[i]=cols[i+1]

    val = np.zeros(3,np.uint8)
    for i in range(3):
        val[i] = int((x*cols[0][i] + (1-x)*cols[1][i])*y+(x*cols[2][i] + (1-x)*cols[3][i])*(1-y))
    return val

def fixoutlines(img,mask,threshold = 10):
    height = len(img)
    width = len(img[0])//2
    with tqdm(total=height*width) as bar:
        for h in range(height):
            for w in range(width):
                bar.update(1)
                px = img[h][w+width]
                if gray(px[0],px[1],px[2]) < threshold:
                    img[h][w] = img[h][w+width]
                px = mask[h][w]
                if px < threshold:
                    col = img[h][w]
                    img[h][w] = [19,7,3]
                    img[h][w+width] = [19,7,3]
    return img
def removeoutlines(base,filtered,masks=[]):
    height = len(base)
    width = len(base[0])//2
    with tqdm(total=height*width) as bar:
        for h in range(height):
            for w in range(width):
                bar.update(1)
                px = filtered[h][w+width]
                if px<200:
                    base[h][w] = [19,7,3]
                    base[h][w+width] = [19,7,3]
                if len(masks)==0:
                    continue
                Max = 0
                for mask in masks:
                    px = mask[h][w]
                    if px>Max:
                        Max = px
                if Max<200:
                    base[h][w] = [19,7,3]
                    base[h][w+width] = [19,7,3]
    return base
def gray(B,G,R):
    return 0.299*R + 0.587*G + 0.114*B
def grayarr(arr):
    return 0.299*arr[0] + 0.587*arr[1] + 0.114*arr[2]

def manipulations(base,angles,reflects,removals,masks,origins,poss):
    for i in range(len(removals)):
        if removals[i]==True:
            base,masks[i],poss[i] = remove(base,masks[i],poss[i])
    for i in range(len(reflects)):
        if removals[i]==True:
            continue
        if reflects[i]:
            base,masks[i],poss[i] = reflect(base,masks[i],origins[i][0],poss[i])
    for i in range(len(angles)):
        if removals[i]==True or angles[i] == 0:
            continue
        base,masks[i],poss[i] = rotate(base,masks[i],angles[i],origins[i],poss[i])
    return base,masks,poss
       
def poly(num,base):
    if num == 0:
        return ["0"]
    ans = []
    while num > 0:
        ans = [str(num%base)] + ans
        num //= base
    return ans

def getconfig(i,angs,Len=4):
    res = []
    r = poly(i,len(angs)*2+1)
    r = ['0']*(Len-len(r))+r
    while(len(r)>Len):
        r = r[1:]

    for c in r:
        if c =='0':
            res.append(None)
        elif int(c)<7:
            res.append([angs[int(c)-1],False])
        else:
            res.append([angs[int(c)-7],True])

    return res

def getCounter(dir):
    files = os.listdir(dir)
    Max = 0
    for file in files:
        num = int(file[:-1])
        if num>Max:
            Max = num
    return Max
def fillocean(base,oceanmask):
    height = len(base)
    width = len(base[0])//2
    with tqdm(total=height*width) as bar:
        for h in range(height):
            for w in range(width):
                bar.update(1)
                px = oceanmask[h][w]
                if px>200:
                    base[h][w] = [19,7,3]
                    base[h][w+width] = [19,7,3]
    return base

def check_empty(im: np.ndarray, coverage: float=0.5) -> bool:
    '''
    Check that the percentage of empty pixels is below the coverage threshold.
    Args:
        im (np.ndarray): The image to check.
        coverage (float): The coverage threshold.
    Returns:
        bool: True if the image is empty, False otherwise.
    '''
    assert im.size > 0
    return np.sum(im == 0) / im.size >= coverage

def save_images(im: np.ndarray, dir: str, name: str, zoom: int=1, full: bool=False, planet_cfg: PlanetConfig=None):
    if full:
        img.fromarray(im).save(os.path.join(dir, '_full', f'{name}.png'))
    sketch, dem = split(im)
    coverage = 1.0 - zoom/7
    v_tiles = 2**zoom
    h_tiles = 2*v_tiles
    sketch_size = 128 if planet_cfg and 'upscaling' in planet_cfg.image_mode  else 256
    dem_size = 256
    for x in range(0, h_tiles):
        for y in range(0, v_tiles):
            sketch_tile = sketch[y*sketch_size:(y+1)*sketch_size, x*sketch_size:(x+1)*sketch_size]
            if planet_cfg and 'inpainting' in planet_cfg.image_mode: 
                sketch_tile = random_masking(sketch_tile)          
            dem_tile = dem[y*dem_size:(y+1)*dem_size, x*dem_size:(x+1)*dem_size]
            if check_empty(dem_tile, coverage):
                tile_dir = os.path.join(dir, '_empty', f'{name}_{y}_{x}/')
            else:
                tile_dir = os.path.join(dir, f'{name}_{y}_{x}/')
            if not os.path.exists(tile_dir):
                os.makedirs(tile_dir)
            img.fromarray(sketch_tile).save(os.path.join(tile_dir, 'sketch.png'))
            img.fromarray(dem_tile).save(os.path.join(tile_dir, 'dem.png'))
            meta = {"range": 84375.0/(2**zoom), "zoom": zoom, "tile_y": y, "tile_x": x, "factor": zoom}
            with open(os.path.join(tile_dir, 'metadata.json'), 'w') as f:
                json.dump(meta, f)


def get_masks(mask_dir, json_dir, num=6):
    masks = []
    positions = []
    origins = []
    for i in range(num):
        mask = np.array(img.open(os.path.join(mask_dir, f"{i}.png")))
        masks.append(mask)
        pos = getPos(masks[i])
        positions.append(pos)
        cx = (np.max(pos[:,1]) + np.min(pos[:,1]))//2
        cy = (np.max(pos[:,0]) + np.min(pos[:,0]))//2
        origins.append([cx, cy])
    return masks, positions, origins


def main():
    #0.299R + 0.587G + 0.114B
    #g = [75,111,147,183]
    #cols = [44,65,86,107,0,255]
    greens = [32,64,96,128,160]
    cols = []
    for g in greens:
        cols.append(0.587*g)
    cols.append(255)
    cols.append(0)
    maincols = cols

    if not os.path.exists(cfg.DATADIR):
        os.makedirs(cfg.DATADIR)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    if not os.path.exists(cfg.TRAIN_DIR):
        os.makedirs(cfg.TRAIN_DIR)
    if not os.path.exists(cfg.TEST_DIR):
        os.makedirs(cfg.TEST_DIR)
    if not os.path.exists(cfg.VALID_DIR):
        os.makedirs(cfg.VALID_DIR)

    if cfg.RESIZE:
        World = cv2.imread('data/manipulations/Merged.jpg')
        width = 2048#int(World.shape[1] * 0.1)
        height = 512#int(World.shape[0] * 0.1)
        Resized = cv2.resize(World, (width,height))
        cv2.imwrite('data/manipulations/{}.jpg'.format("Merged"), Resized)
        World = cv2.imread('data/manipulations/Coloured.jpg')
        Resized = cv2.resize(World, (width,height))
        cv2.imwrite('data/manipulations/{}.jpg'.format("Coloured"), Resized)
        

    if cfg.FILTER:
        merged = cv2.imread('data/manipulations/BaseFixed.jpg',0)
        filtered = filter(merged,10)
        filtered = removescraps(filtered)
        cv2.imwrite('data/manipulations/{}.jpg'.format("Filtered"), filtered)
    if cfg.FIXOUTLINES:
        basefixed = cv2.imread('data/manipulations/4320_1080/BaseFixed.jpg')
        #mask = cv2.imread('data/manipulations/Filtered.jpg',0)
        outlines,r = split(cv2.imread('data/manipulations/4320_1080/Remainder.jpg',0))
        pos = getPos(outlines)
        #l,r = split(mask)
        #fixedoutlines = fixoutlines(merged,r,10)
        fixedoutlines,outlines,pos = remove(basefixed,outlines,pos)
        cv2.imwrite('data/manipulations/4320_1080/{}.jpg'.format("Base"), fixedoutlines)
        # filtered = cv2.imread('data/manipulations/Filtered.jpg')
        # fixedoutlines = fixoutlines(filtered,r,10)
        # cv2.imwrite('data/manipulations/{}.jpg'.format("Filtered"), fixedoutlines)
    #img = normalize(r,cols)
    if cfg.GETOBJECTS:
        colours = cv2.imread('data/manipulations/Coloured.jpg',0)
        l,r = split(colours)
        objs = getobjects(l,maincols)
        i=0
        for obj in objs:
            #x,y = getcenter(obj)
            #obj = removeoutliers(x,y,400,obj)
            with tqdm(total=len(obj)) as bar:
                img = np.zeros((len(l),len(l[0]),1), np.uint8)
                for pos in obj:
                    img[pos[0]][pos[1]] = 255
                    bar.update(1)
                #img = removescraps(img)
                cv2.imwrite('data/manipulations/{}.jpg'.format(i), img)
                i+=1
        cv2.imwrite('data/manipulations/Objects.jpg', l)
    if cfg.GETREMAINDER:
        masks, poss, _ = get_masks(6)
        filtered = cv2.imread('data/manipulations/4320_1080/Base.jpg')
        print(len(filtered[0]))
        for i in range(6):
            filtered,masks[i],poss[i] = remove(filtered,masks[i],poss[i])
            print(len(masks[i][0]))
        cv2.imwrite('data/manipulations/4320_1080/Remainder.jpg', filtered)

        
    if cfg.REFLECT:
        # base = cv2.imread('data/manipulations/BaseFixed.jpg')
        # mask = cv2.imread('data/manipulations/0.jpg',0)
        # axis = getXaxis(mask)
        # out,mask = reflect(base,mask,axis)
        # cv2.imwrite('data/manipulations/reflections/test.jpg', out)
        # exit(0)
        masks = []
        axises = []
        poss = []
        for i in range(4):
            masks.append(cv2.imread('data/manipulations/{}.jpg'.format(i),0))
            poss.append(getPos(masks[i]))
            axises.append(getXaxis(masks[i],poss[i]))
        #for i in range(1,17):
        
        for i in range(1,16):
            base = cv2.imread('data/manipulations/BaseFixed.jpg')
            selection = bin(i)
            selection = selection[selection.index('b')+1:]
            
            out = compoundreflect(base,masks,axises,poss,selection)
            cv2.imwrite('data/manipulations/reflections/{}.jpg'.format(selection), out)
    if cfg.FILLOCEAN:
        #mask = cv2.imread('data/Filtered.png',0)
        oceanmask = cv2.imread('data/manipulations/6.jpg',0)
        # masks = []
        # for i in range(6):
        #     masks.append(cv2.imread('data/manipulations/{}.jpg'.format(i),0))
        #     print(getvals(masks[i]))
        img = cv2.imread('data/manipulations/BaseFixed.jpg')
        img = fillocean(img,oceanmask)
        #img = removeoutlines(img,mask,masks)
        
        cv2.imwrite('data/manipulations/BaseFixed.jpg', img)
    if cfg.TRANSLATE:
         mask = cv2.imread('data/manipulations/2.jpg',0)
         base = cv2.imread('data/manipulations/BaseFixed.jpg')
         base,mask = translate(base,mask,-100,0)
         #base,mask = translate(base,mask,-100,0)
         #base, mask = reflect(base,mask)
         cv2.imwrite('data/manipulations/Translated.jpg', base)
    if cfg.ROTATE:
        # mask = cv2.imread('data/manipulations/2.jpg',0)
        # base = cv2.imread('data/manipulations/BaseFixed.jpg')
        # for i in range(0,360,30):
        #     base,mask = rotate(base,mask,30)
        #     cv2.imwrite('data/manipulations/rotations/Rotated{}.jpg'.format(i), base)
        #     cv2.imwrite('data/manipulations/rotations/RotatedMask{}.jpg'.format(i), mask)
        for j in range(4):
            mask = cv2.imread('data/manipulations/{}.jpg'.format(j),0)
            pos = getPos(mask)
            origin = [getXaxis(mask,pos),getYaxis(mask,pos)]
            for i in range(-30,35,5):
                if i==0:
                    continue
                mask = cv2.imread('data/manipulations/{}.jpg'.format(j),0)
                base = cv2.imread('data/manipulations/BaseFixed.jpg')
                base,mask,Pos = rotate(base,mask,i,origin,pos)
                cv2.imwrite('data/manipulations/rotations/{}Rotated{}.jpg'.format(j,i), base)
            for i in range(-30,35,5):
                mask = cv2.imread('data/manipulations/{}.jpg'.format(j),0)
                base = cv2.imread('data/manipulations/BaseFixed.jpg')
                base,mask,Pos = rotate(base,mask,i+180,origin,pos)
                cv2.imwrite('data/manipulations/rotations/{}Rotated{}.jpg'.format(j,i+180), base)
                #cv2.imwrite('data/manipulations/rotations/{}RotatedMask{}.jpg'.format(j,i), mask)
    if cfg.REMOVE:
        for i in range(6):
            mask = cv2.imread('data/manipulations/{}.jpg'.format(i),0)
            base = cv2.imread('data/manipulations/BaseFixed.jpg')
            base,mask = remove(base,mask)
            cv2.imwrite('data/manipulations/removals/Removal{}.jpg'.format(i), base)
    if cfg.ALL:
        masks = []
        origins = []
        poss = []
        t = time()
        for i in range(6):
            mask_dir = os.path.join(cfg.DATADIR, 'manipulations/4320_1080/{}.jpg'.format(i))
            
            mask =cv2.imread(mask_dir, 0)
            masks.append(cv2.resize(mask, (cfg.WIDTH, cfg.HEIGHT)))
            poss.append(getPos(masks[i]))
            origins.append([getXaxis(masks[i],poss[i]),getYaxis(masks[i],poss[i])])
        base = cv2.resize(cv2.imread(os.path.join(cfg.DATADIR, 'manipulations/4320_1080/Base.jpg')), (cfg.WIDTH*2, cfg.HEIGHT))
        print("Initialisation done in {} seconds".format(time()-t))
        refs = [False,True]
        angs = [0,-15,15,165,180,195]
        rems = [False,True]
        used = 4
        m = (len(angs)*2+1)**(used)
        
        count = getCounter(cfg.TRAIN_DIR)
        generated = [False]*m
        try:
            with open(os.path.join(cfg.OUTPUT_DIR, 'generated.txt'),'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    generated[i] = True if line.strip()=='True' else False 
        except:
            pass
        cnt = count
        #2*(m-count)*2**(len(masks)-used)
        with tqdm(total=(cfg.ITERS)*4-cnt) as bar:
            for i in range((cnt+1)//4, cfg.ITERS):
                for b1 in rems:
                    for b2 in rems:
                        r = random.randint(0,m-1)
                        while generated[r] and False in generated:
                            r = random.randint(0,m-1)
                        config = getconfig(r,angs)
                        generated[r] = True
                        angles = []
                        reflects = []
                        removals = []
                        for c in config:
                            if not c:
                                angles.append(0)
                                reflects.append(False)
                                removals.append(True)
                            else:
                                angles.append(c[0])
                                reflects.append(c[1])
                                removals.append(False)
                        for k in range(len(masks)-used):
                            removals.append(False)
                        removals[used] = random.randint(0,1) == 0
                        removals[used+1] = random.randint(0,1) == 0
                        Base = np.ndarray.copy(base)
                        Masks = []
                        for mask in masks:
                            Masks.append(np.ndarray.copy(mask))
                        Poss = []
                        for pos in poss:
                            temp = []
                            for p in pos:
                                temp.append([p[0],p[1]])
                            Poss.append(temp)
                        
                        B,M,P = manipulations(Base,angles.copy(),reflects.copy(),removals.copy(),Masks,origins,Poss)
                        # cv2.imwrite('data/train/{}.jpg'.format(count), B)
                        # bar.update(1)
                        # count+=1
                        B1,B2 = split(B)
                        Spin = random.randint(50,len(B1[0])-50)
                        B1 = spin(B1,Spin)
                        B2 = spin(B2,Spin)
                        B = np.concatenate((B1,B2), axis = 1)
                        if b1:
                            B = cv2.flip(B, 0)
                        # cv2.imwrite(os.path.join(cfg.TRAIN_DIR, '{}.jpg'.format(count)) , B)
                        save_images(B, cfg.TRAIN_DIR, r)
                        bar.update(1)
                        count+=1
        count = getCounter(cfg.VALID_DIR)
        with tqdm(total=cfg.ITERS*4//10) as bar:
            for i in range(cfg.ITERS//10):
                for b1 in rems:
                    for b2 in rems:
                        r = random.randint(0,m-1)
                        while generated[r]:
                            r = random.randint(0,m-1)
                        config = getconfig(r,angs)
                        generated[r] = True
                        angles = []
                        reflects = []
                        removals = []
                        for c in config:
                            if not c:
                                angles.append(0)
                                reflects.append(False)
                                removals.append(True)
                            else:
                                angles.append(c[0])
                                reflects.append(c[1])
                                removals.append(False)
                        for k in range(len(masks)-used):
                            removals.append(False)
                        removals[used] = random.randint(0,1) == 0
                        removals[used+1] = random.randint(0,1) == 0
                        Base = np.ndarray.copy(base)
                        Masks = []
                        for mask in masks:
                            Masks.append(np.ndarray.copy(mask))
                        Poss = []
                        for pos in poss:
                            temp = []
                            for p in pos:
                                temp.append([p[0],p[1]])
                            Poss.append(temp)
                        
                        B,M,P = manipulations(Base,angles.copy(),reflects.copy(),removals.copy(),Masks,origins,Poss)
                        # cv2.imwrite('data/train/{}.jpg'.format(count), B)
                        # bar.update(1)
                        # count+=1
                        B1,B2 = split(B)
                        Spin = random.randint(50,len(B1[0])-50)
                        B1 = spin(B1,Spin)
                        B2 = spin(B2,Spin)
                        B = np.concatenate((B1,B2), axis = 1)
                        if b1:
                            B = cv2.flip(B, 0)
                        # cv2.imwrite(os.path.join(cfg.VALID_DIR, '{}.jpg'.format(count)) , B)
                        save_images(B, cfg.VALID_DIR, count)
                        bar.update(1)
                        count+=1  
        count = getCounter(cfg.TEST_DIR) 
        with tqdm(total=cfg.ITERS*4//10) as bar:
            for i in range(cfg.ITERS//10):
                for b1 in rems:
                    for b2 in rems:
                        r = random.randint(0,m-1)
                        while generated[r]:
                            r = random.randint(0,m-1)
                        config = getconfig(r,angs)
                        generated[r] = True
                        angles = []
                        reflects = []
                        removals = []
                        for c in config:
                            if not c:
                                angles.append(0)
                                reflects.append(False)
                                removals.append(True)
                            else:
                                angles.append(c[0])
                                reflects.append(c[1])
                                removals.append(False)
                        for k in range(len(masks)-used):
                            removals.append(False)
                        removals[used] = random.randint(0,1) == 0
                        removals[used+1] = random.randint(0,1) == 0
                        Base = np.ndarray.copy(base)
                        Masks = []
                        for mask in masks:
                            Masks.append(np.ndarray.copy(mask))
                        Poss = []
                        for pos in poss:
                            temp = []
                            for p in pos:
                                temp.append([p[0],p[1]])
                            Poss.append(temp)
                        
                        B,M,P = manipulations(Base,angles.copy(),reflects.copy(),removals.copy(),Masks,origins,Poss)
                        # cv2.imwrite('data/train/{}.jpg'.format(count), B)
                        # bar.update(1)
                        # count+=1
                        B1,B2 = split(B)
                        Spin = random.randint(50,len(B1[0])-50)
                        B1 = spin(B1,Spin)
                        B2 = spin(B2,Spin)
                        B = np.concatenate((B1,B2), axis = 1)
                        if b1:
                            B = cv2.flip(B, 0)
                        # cv2.imwrite(os.path.join(cfg.TEST_DIR, '{}.jpg'.format(count)) , B)
                        save_images(B, cfg.TEST_DIR, count)
                        bar.update(1)
                        count+=1        
        with open(os.path.join(cfg.OUTPUT_DIR, 'generated.txt'),'w') as f:
            for gen in generated:
                f.write(str(gen)+'\n')
        
        print('Saved')
    if cfg.RENAME:
        dir = 'data/manipulations/compound/'
        files = os.listdir(dir)
        for file in files:
            #print(file)
            os.rename(dir+file,dir+file+'.jpg')
    if cfg.OTHER:
        basefixed = cv2.imread('data/manipulations/BaseFixed.jpg')
        l,r = split(basefixed)
        width = 2048
        height = 1024
        l = cv2.resize(l,(width,height))
        r = cv2.imread('data/manipulations/4320_1080/Resized.jpg') 
        r = cv2.resize(r,(width,height))
        basefixed = np.concatenate((l,r), axis = 1)
        cv2.imwrite('data/manipulations/BaseFixed.jpg',basefixed)   
        for i in range(7):
            img = cv2.imread(f'data/manipulations/{i}.jpg')
            img = cv2.resize(img,(width,height))
            cv2.imwrite(f'data/manipulations/{i}.jpg',img)
    if cfg.GENERATE:
        
        masks = []
        origins = []
        poss = []
        t = time()
        operations = {
            'ref': [[False,True]]*4 + [[False]]*2,
            'ang': [[0,-15,15,165,180,195]]*4 + [[0]]*2,
            'rem': [[False,True]]*6,
        }
        
        masks, poss, origins = get_masks()
        base = cv2.resize(cv2.imread(os.path.join(cfg.DATADIR, 'manipulations/4320_1080/Base.jpg')), (cfg.WIDTH*2, cfg.HEIGHT))
        print("Initialisation done in {} seconds".format(time()-t))
        refs = [False,True]
        angs = [0,-15,15,165,180,195]
        rems = [False,True]
        used = 4
        m = 1327104 # 2*2*2*2*1*1*6*6*6*6*1*1*2*2*2*2*2*2
        m *= 2 # horizontal flip
        m *= 2 # vertical flip
        generated = [False]*m
        try:
            with open(os.path.join(cfg.OUTPUT_DIR, 'generated.txt'),'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    generated[int(line.strip())] = True 
        except:
            pass
        for d, iters in [(cfg.TRAIN_DIR, cfg.ITERS), (cfg.TEST_DIR, cfg.ITERS//10), (cfg.VALID_DIR, cfg.ITERS//10)]:
            cnt = len(os.listdir(d))//2
            with tqdm(total=iters-cnt) as bar:
                for i in range((cnt+1), iters):
                    k = random.randint(0,m-1)
                    name = k
                    while generated[k] and False in generated:
                        k = random.randint(0,m-1)
                    mask_refs = [False]*6
                    mask_angs = [0]*6
                    mask_rems = [False]*6
                    for i in range(4):
                        mask_refs[i] = operations['ref'][i][k%2]
                        k//=2
                    for i in range(4):
                        mask_angs[i] = operations['ang'][i][k%6]
                        k//=2
                    for i in range(6):
                        mask_rems[i] = operations['rem'][i][k%2]
                        k//=2
                    
                    generated[k] = True
                    Base = np.ndarray.copy(base)
                    Masks = []
                    for mask in masks:
                        Masks.append(np.ndarray.copy(mask))
                    Poss = []
                    for pos in poss:
                        temp = []
                        for p in pos:
                            temp.append([p[0],p[1]])
                        Poss.append(temp)
                    
                    B,M,P = manipulations(Base, mask_angs, mask_refs, mask_rems, Masks, origins, Poss)
                    # cv2.imwrite('data/train/{}.jpg'.format(count), B)
                    # bar.update(1)
                    # count+=1
                    B1,B2 = split(B)
                    if k%2 == 1:
                        B1 = cv2.flip(B1, 1)
                        B2 = cv2.flip(B2, 1)
                    k//=2
                    Spin = random.randint(50,len(B1[0])-50)
                    B1 = spin(B1,Spin)
                    B2 = spin(B2,Spin)
                    B = np.concatenate((B1,B2), axis = 1)
                    if k%2 == 1:
                        B = cv2.flip(B, 0)
                    k//=2
                    # cv2.imwrite(os.path.join(cfg.TRAIN_DIR, '{}.jpg'.format(count)) , B)
                    save_images(B, d, name)
                    with open(os.path.join(cfg.OUTPUT_DIR, 'generated.txt'),'a') as f:
                        f.write(str(name)+'\n')
                    bar.update(1)
         
                    
        
        print('Saved')


if __name__ == '__main__':
    main()

# img = cv2.imread('data/0.jpg')
# img0,img1 = split(img)
# for i in range(1,2*len(img)//10):
#     horizontal = np.concatenate((rotate(img0,i*10), rotate(img1,i*10)), axis = 1)
#     cv2.imwrite('data/train/{}.jpg'.format(i), horizontal)
# img = cv2.imread('data/0.jpg',0)
# height = len(img)
# width = len(img[0])
# print(height,width)
# cv2.imwrite('data/img0.jpg', rotate(img0,100))
# cv2.imwrite('data/img1.jpg', rotate(img1,100))
# filtered = filter(img,20)
# edge = edges(filtered,1,2)
# cv2.imwrite('data/Edges.jpg', edge)
# cv2.imwrite('data/Filtered.jpg', filtered)