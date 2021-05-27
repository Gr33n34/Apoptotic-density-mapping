# Importing modules
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
from google.colab import files
import scipy.stats as st
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from PIL import Image, ImageFilter


# Writing a function that expands result matrix to image dimensions
def writing_data_to_matrix(datamatrix, image):
  x_len=image.shape[0]/datamatrix.shape[0]
  y_len=image.shape[1]/datamatrix.shape[1]
  empty_matrix = np.repeat(datamatrix, x_len, axis=0)
  empty_matrix = np.repeat(empty_matrix, y_len, axis=1)
  return empty_matrix


# Defining kernel of 2D-gaussian distribution and normalizing to pixel value
def gkern(kernlen=21, nsig=3):
  x = np.linspace(-nsig, nsig,kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kern2d = np.outer(kern1d, kern1d)
  return kern2d/kern2d.sum()


# Applying the kernel to the created result matrix
def apply_kernel(input, kernlen=64, sigma=2.5, iter=1):
  kernel = gkern(kernlen, sigma)*273
  kernel = kernel / np.sum(kernel[:])
  for i in range(iter):
    input = sp.ndimage.filters.convolve(input, kernel, mode='constant')
    print(i)
  return input

# Function that takes normalized matrix and returns image in red gradation
def writing_to_red_img(np_array):
  # Calibrating normalized data to max pixel value
  data = np_array*255
  # Getting mock lattice
  img_lattice=np.zeros((data.shape[0], data.shape[1], 3))
  # Writing pixel values to red color shades
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      pxval=255-data[i][j]
      img_lattice[i][j][0]=255
      img_lattice[i][j][1]=pxval
      img_lattice[i][j][2]=pxval
  # Writing to RGB image
  PIL_image = Image.fromarray(np.uint8(img_lattice)).convert('RGB')
  return PIL_image


# Importing image data
uploaded = files.upload()
img = cv2.imread("merged_sample_image.jpg",1)
#Getting blue channel for trabecular architecture
blue_channel = img[:,:,0]
# Cropping image to desired dimensions
img_cropped = blue_channel[:560,:700]
# Gaussian filtering and Otsu thresholding for boolean image
blur = cv2.GaussianBlur(blue_channel,(5,5),0)
threshold,img_bool= cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Importing normalized cell count data
cc_data = pd.read_excel("Cell_count_matrix.xlsx", header=None, sheet_name="Sheet1").to_numpy()
# Expanding data to image dimensions
cc_data_exp = writing_data_to_matrix(cc_data,img_bool)
# Creating stochastic distribution map
cc_data_stoch = apply_kernel(cc_data_exp,iter=4,kernlen=100,sigma=1)
# Apply boolean image
img_mapped = cc_data_stoch * img_bool
# Creating image file and export
output = writing_to_red_img(img_mapped)
output.save('mapped_sample_image.png')
