import numpy as np
import cv2
from os.path import dirname, join, basename
from glob import glob

bin_n = 16 * 16 # Number of bins

def hog(img):
	x_size = 194;
	y_size = 259;
	sobelX = cv2.Sobel(img, cv2.CV_32F, 1, 0);
	sobelY = cv2.Sobel(img, cv2.CV_32F, 0, 1);
	#print sobelX, sobelY;
	mag, ang = cv2.cartToPolar(sobelX, sobelY);
	angsInDegree = np.int32(bin_n * ang / (2 * np.pi));
	magFourBlocks = [mag[:x_size / 2, :y_size / 2], mag[x_size / 2:, :y_size / 2], mag[:x_size / 2, y_size / 2:], mag[x_size / 2:, y_size / 2:]];
	angFourBlocks = [angsInDegree[:x_size / 2, :y_size / 2], angsInDegree[x_size / 2:, :y_size / 2], angsInDegree[:x_size / 2, y_size / 2:], angsInDegree[x_size / 2:, y_size / 2:]];
	# count the occurrences
	occur = [np.bincount(i.ravel(), j.ravel(), bin_n) for i, j in zip(angFourBlocks, magFourBlocks)];

	return np.hstack(occur);

img = [];
for i in glob(join(dirname(__file__) + '/cat', '*.jpg')):
	img.append(cv2.imread(i, 0));
catNum = len(img);
for i in glob(join(dirname(__file__) + '/other', '*.jpg')):
	img.append(cv2.imread(i, 0));
hogData = np.float32(map(hog, img)).reshape(-1, bin_n * 4);
dataResult = np.float32(np.repeat(1.0, len(img))).reshape(len(img), 1);
dataResult[catNum : len(img)] = 0;

svmParams = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 2.67, gamma = 5.383);
#print dataResult;
#print np.float32(hogData).shape;
svm = cv2.SVM();
svm.train(hogData, dataResult, params = svmParams);
svm.save("svm_cat_data.dat");


