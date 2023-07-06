import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("TrackBar")
#cv2.resizeWindow("TrackBar", 500, 320)

cv2.createTrackbar("Lower - H", "TrackBar", 0, 180, nothing)
cv2.createTrackbar("Lower - S", "TrackBar", 0, 255, nothing)
cv2.createTrackbar("Lower - V", "TrackBar", 0, 255, nothing)

cv2.createTrackbar("Upper - H", "TrackBar", 0, 180, nothing)
cv2.createTrackbar("Upper - S", "TrackBar", 0, 255, nothing)
cv2.createTrackbar("Upper - V", "TrackBar", 0, 255, nothing)

cv2.setTrackbarPos("Upper - H", "TrackBar", 255)
cv2.setTrackbarPos("Upper - S", "TrackBar", 255)
cv2.setTrackbarPos("Upper - V", "TrackBar", 255)

while True:
    frame = cv2.imread('rail1.jpeg')
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_h = cv2.getTrackbarPos("Lower - H", "TrackBar")
    lower_s = cv2.getTrackbarPos("Lower - S", "TrackBar")
    lower_v = cv2.getTrackbarPos("Lower - V", "TrackBar")

    upper_h = cv2.getTrackbarPos("Upper - H", "TrackBar")
    upper_s = cv2.getTrackbarPos("Upper - S", "TrackBar")
    upper_v = cv2.getTrackbarPos("Upper - V", "TrackBar")

    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(frame_hsv, lower_color, upper_color)
    #cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Original', 480, 480)
    cv2.imshow("Original", frame)
    #cv2.namedWindow('Masked', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Masked', 480, 480)
    cv2.imshow("Masked", mask)

    if cv2.waitKey(25) == 27:
        break

cv2.destroyAllWindows()