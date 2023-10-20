import cv2
import os

os.environ['QT_STYLE_OVERRIDE'] = 'xcb'

img = cv2.imread('train/2ca0d812-IMG_8218.jpg')
cv2.imshow("my_window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()