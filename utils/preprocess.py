import cv2 as cv
import nibabel as nib
import numpy as np
from deepbrain import Extractor

def get_data(path):
    return nib.load(path).get_data()

def get_data_with_skull_scraping(path, PROB = 0.5):
    arr = nib.load(path).get_data()
    ext = Extractor()
    prob = ext.run(arr)
    mask = prob > PROB
    arr = arr*mask
    return arr

def histeq(data):
    for slice_index in range(data.shape[2]):
        data[:,:,slice_index]=cv.equalizeHist(data[:,:,slice_index])
    return data

def to_uint8(data):
    data=data.astype(np.float)
    data[data<0]=0
    return ((data-data.min())*255.0/data.max()).astype(np.uint8)

def IR_to_uint8(data):
    data=data.astype(np.float)
    data[data<0]=0
    return ((data-800)*255.0/data.max()).astype(np.uint8)
