import cv2
import numpy as np

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    print(type(frame))
  
    # Display the resulting frame
    frame = cv2.Canny(frame,100,200)
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

