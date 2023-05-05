import subprocess
import cv2
import matplotlib.pyplot as plt

# vid = cv2.VideoCapture(0)
# ret, frame = vid.read()

# cv2.imshow('frame', frame)
# # cv2.waitKey(0)

# filename = r"D:\Work\UTA\Academics\Spring 2023\CSE 6367 - Computer Vision\Project\Repositories\monodepth2\assets\code_tester.jpg"

# cv2.imwrite(filename, frame)

depth = cv2.imread(r"D:\Work\UTA\Academics\Spring 2023\CSE 6367 - Computer Vision\Project\Repositories\monodepth2\assets\code_tester_disp.jpeg") 
main = cv2.imread(r"D:\Work\UTA\Academics\Spring 2023\CSE 6367 - Computer Vision\Project\Repositories\monodepth2\assets\code_tester.jpg") 
grey_depth  = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(main,100,200)

# for x in range(grey_depth.shape[0]):
#     for y in range(grey_depth.shape[1]):
#         if grey_depth[x,y] > 150:
#             grey_depth[x,y] = 1
#         else:
#             grey_depth[x,y] = 0

grey_depth = [grey_depth > 170]

print(grey_depth)

main[:,:,0] = main[:,:,0] * grey_depth
main[:,:,1] = main[:,:,1] * grey_depth
main[:,:,2] = main[:,:,2] * grey_depth


cv2.imshow('frame',main)
cv2.waitKey(0)