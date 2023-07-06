import cv2
import numpy as np

kernel1 = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])

img = cv2.imread("rail1.jpeg")

img = cv2.pyrMeanShiftFiltering(img, 20, 20)
img = cv2.filter2D(img, -1, kernel1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)
lines = cv2.HoughLines(edges, 1, np.pi/180, 250)

blank = np.zeros_like(img)
blank[edges != 0] = (255, 255, 255)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        x0 = costheta * rho
        y0 = sintheta * rho
        x1 = int(x0 - rho * (-sintheta))
        y1 = int(y0 + rho * costheta)
        x2 = int(x0 + rho * (-sintheta))
        y2 = int(y0 - rho * costheta)
        if rho > 100:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)


cv2.imshow("bra",blank)
cv2.waitKey(0)
cv2.imshow("bra2",img)
cv2.waitKey(0)