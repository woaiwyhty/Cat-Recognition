import numpy as np
import cv2
from os.path import dirname, join, basename
from glob import glob

num = 0;
for fn in glob(join(dirname(__file__) + "other", "*.jpg")):
	mImage = cv2.imread(fn);
	res = cv2.resize(mImage, (64, 128), interpolation = cv2.INTER_AREA);
	cv2.imwrite("ResizedPics/" + str(num) + ".jpg", res);
	num = num + 1;

print("Done!");


cv2.waitKey(0)
cv2.destroyAllWindows()
