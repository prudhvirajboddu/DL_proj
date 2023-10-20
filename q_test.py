import cv2 as cv
img = cv.imread("/workspaces/DL_proj/1c620bed-IMG_8229.jpg")

cv.imshow("Display window", img)
k = cv.waitKey(0)
# print(cv.getBuildInformation())