import cv2
import numpy as np

kernel = np.ones((5,5))
kernel2 = np.ones((2,5))
lower_hsv = np.array([0,30,30])
upper_hsv = np.array([100,255,255])


def detectCorner(image:np.array):
    """
    Detects corners in an image using the Canny edge detection algorithm.

    Parameters:
    - image (numpy.ndarray): The input image on which the corner detection will be performed. 
      It should be a 3-channel (BGR) or a single-channel (grayscale) image.

    Returns:
    - edges (numpy.ndarray): A binary image representing the detected edges.
    """
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayImage, 100, 150)
    return edges


def hsvDetection(image:np.array,lower:np.array,upper:np.array):
    """
    Performs HSV color detection on an image and returns a binary mask.

    Parameters:
    - image (numpy.ndarray): The input image on which the color detection will be performed.
      It should be a 3-channel (BGR) image.
    - lower (numpy.ndarray): The lower bound of the HSV color range to detect.
      It should be a 1-dimensional array of shape (3,) representing the lower values for hue, saturation, and value.
    - upper (numpy.ndarray): The upper bound of the HSV color range to detect.
      It should be a 1-dimensional array of shape (3,) representing the upper values for hue, saturation, and value.

    Returns:
    - mask (numpy.ndarray): A binary mask representing the pixels within the specified HSV color range.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask


def resize_image(image:np.array, width):
    """
    Resizes an image while maintaining its aspect ratio.

    Parameters:
    - image (numpy.ndarray): The input image to be resized.
    - width (int): The desired width of the resized image.

    Returns:
    - resized_image (numpy.ndarray): The resized image with the specified width.
    """
    ratio = width / image.shape[1]
    height = int(image.shape[0] * ratio)
    new_size = (width, height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


def draw_best_fit_line(image:np.array, contours:list):
    """
    Draws the best-fit line through a set of contours on an image.

    Parameters:
    - image (numpy.ndarray): The input image on which the line will be drawn.
    - contours (list): A list of contours representing the points used to compute the best-fit line.

    Returns:
    - image (numpy.ndarray): The input image with the best-fit line drawn on it.
    - pt1 (tuple): The coordinates of the first endpoint of the best-fit line.
    - pt2 (tuple): The coordinates of the second endpoint of the best-fit line.
    """
    points = np.vstack(contours).squeeze()
    vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-cx * vy / vx) + cy)
    righty = int(((image.shape[1] - cx) * vy / vx) + cy)
    pt1 = (image.shape[1] - 1, righty)
    pt2 = (0, lefty)
    cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    return image,pt1,pt2


def find_intersecting_contours(line_start:tuple, line_end:tuple, contours:list):
    """
    Finds the contours that intersect with a given line segment.

    Parameters:
    - line_start (tuple): The coordinates of the starting point of the line segment.
    - line_end (tuple): The coordinates of the ending point of the line segment.
    - contours (list): A list of contours to be checked for intersection.

    Returns:
    - intersecting_contours (list): A list of contours that intersect with the line segment.
    """
    intersecting_contours = []
    for contour in contours:
        contour_points = contour[:, 0, :]
        for point in contour_points:
            x, y = point
            distance = cv2.pointPolygonTest(np.array([line_start, line_end]), (int(x), int(y)), True)
            #print(distance)
            if distance < 0 and distance > -10:
                intersecting_contours.append(contour)
                break
    return intersecting_contours


#**********************************************************

image = cv2.imread("rail1.jpeg")
image = resize_image(image,1000)

edges = detectCorner(image)
edges = cv2.dilate(edges,kernel)
edges = cv2.bitwise_not(edges)

masked_image = hsvDetection(image,lower_hsv,upper_hsv)

bitwise_image = cv2.bitwise_and(masked_image,edges)

bitwise_image = cv2.erode(bitwise_image,kernel2,iterations=5)
bitwise_image = cv2.dilate(bitwise_image,kernel,iterations=5)

contours, _ = cv2.findContours(bitwise_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
contour_areas = [cv2.contourArea(contour) for contour in contours]
#print(contour_areas) 
median = np.median(contour_areas)
print("Median:", median)

blank = np.zeros_like(image)
threshold_area = 30000

thresh_contour = []
for contour in contours:
    contour_area = cv2.contourArea(contour)
    if contour_area > threshold_area:
        thresh_contour.append(contour)

for contour in thresh_contour:
    cv2.drawContours(blank, [contour], 0, (255, 255, 255), thickness=1)

blank,pt1,pt2 = draw_best_fit_line(blank,thresh_contour)

intersect_contours = find_intersecting_contours(pt1,pt2,thresh_contour)
#print(len(thresh_contour))
#print(len(intersect_contours))

if len(intersect_contours) >= 2:
    print('Ray hasarlidir')
else:
    print('Ray hasarsizdir')

cv2.imshow("Kenarlar", blank)
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
