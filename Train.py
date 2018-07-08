import numpy as np
import cv2
from os.path import dirname, join, basename
from glob import glob

bin_n = 16 * 16 # Number of bins

def hog(img):
	x_size = 19;
	y_size = 259;
	sobelX = cv2.Sobel(img, cv2.CV_32F, 1, 0);
	sobelY = cv2.Sobel(img, cv2.CV_32F, 0, 1);
	#print sobelX, sobelY;
	mag, ang = cv2.cartToPolar(sobelX, sobelY);
	angsInDegree = np.int32(bin_n * ang / (2 * np.pi));


img = [];
for i in glob(join(dirname(__file__) + 'cat', '*.jpg')):
	img.append(cv2.imread(i, 0));

temp = hog(img[0]);
