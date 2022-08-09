import cv2
import numpy as np

# Load image
image = cv2.imread('Capture.JPG', 0)

#This is for the dirt Image
""" 
Set our filtering parameters
Initialize parameter setting using cv2.SimpleBlobDetector
"""

dirt = cv2.SimpleBlobDetector_Params()
# Set Area filtering parameters
dirt.filterByArea = True
dirt.minArea = 100
# Set Convexity filtering parameters
dirt.filterByConvexity = True
dirt.minConvexity = 0.01
# Set inertia filtering parameters
dirt.filterByInertia = True
dirt.minInertiaRatio = 0.01


#This is for the actual dots
"""
Set our filtering parameters
Initialize parameter setting using cv2.SimpleBlobDetector 
"""

dots = cv2.SimpleBlobDetector_Params()
# Set Area filtering parameters
dots.filterByArea = True
dots.minArea = 45
# Set Circularity filtering parameters
dots.filterByCircularity = True
dots.minCircularity = 0.9
# Set Convexity filtering parameters
dots.filterByConvexity = True
dots.minConvexity = 0.2
# Set inertia filtering parameters
dots.filterByInertia = True
dots.minInertiaRatio = 0.01

# Create a detector with the parameters
dirt_detector = cv2.SimpleBlobDetector_create(dirt)
dot_detector = cv2.SimpleBlobDetector_create(dots)

# Detect blobs
dirt_keypoints = dirt_detector.detect(image)
dot_keypoints = dot_detector.detect(image)

# Draw blobs on our image
blank = np.zeros((1, 1))

dirt_blobs = cv2.drawKeypoints(image, dirt_keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dot_blobs = cv2.drawKeypoints(image, dot_keypoints, blank, (20, 205, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Display Number of detected Dirts
number_of_dirt_blobs = len(dirt_keypoints)
dirt_text = "Number of detected dirts: " + str(len(dirt_keypoints))
cv2.putText(dirt_blobs, dirt_text, (20, 550),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#Display Number of detected Dotts
number_of_dot_blobs = len(dot_keypoints)
dot_text = "Number of correct dots: " + str(len(dot_keypoints))
cv2.putText(dot_blobs, dot_text, (20, 550),cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 205, 0), 2)

quantity_of_dirtDot = number_of_dot_blobs +number_of_dirt_blobs
print("Total Quantity of Dirt and Dots Detected in whole Image: ",quantity_of_dirtDot)
# Show blobs
cv2.imshow("Dirts", dirt_blobs)
cv2.imshow("Dots", dot_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
For Area of Interest and Counting of Dots and Dirts
"""
# Select ROI
region = cv2.selectROI("Please select any area of Interest", image)
# Crop image
cropped_image = image[int(region[1]):int(region[1] + region[3]),int(region[0]):int(region[0] + region[2])]

#This is for the dirt Image
""" 
Set our filtering parameters
Initialize parameter setting using cv2.SimpleBlobDetector
"""
dirt = cv2.SimpleBlobDetector_Params()
# Set Area filtering parameters
dirt.filterByArea = True
dirt.minArea = 100
# Set Convexity filtering parameters
dirt.filterByConvexity = True
dirt.minConvexity = 0.01
# Set inertia filtering parameters
dirt.filterByInertia = True
dirt.minInertiaRatio = 0.01


#This is for the actual dots
""" 
Set our filtering parameters
Initialize parameter setting using cv2.SimpleBlobDetector
"""
dots = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
dots.filterByArea = True
dots.minArea = 45
#
# Set Circularity filtering parameters
dots.filterByCircularity = True
dots.minCircularity = 0.9
# Set Convexity filtering parameters
dots.filterByConvexity = True
dots.minConvexity = 0.2
# Set inertia filtering parameters
dots.filterByInertia = True
dots.minInertiaRatio = 0.01

# Create a detector with the parameters
dirt_detector = cv2.SimpleBlobDetector_create(dirt)
dot_detector = cv2.SimpleBlobDetector_create(dots)

# Detect blobs
dirt_keypoints = dirt_detector.detect(cropped_image)
dot_keypoints = dot_detector.detect(cropped_image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
dirt_blobs = cv2.drawKeypoints(cropped_image, dirt_keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

dot_blobs = cv2.drawKeypoints(cropped_image, dot_keypoints, blank, (20, 205, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#

#Display Number of detected Dirts
number_of_dirt_blobs = len(dirt_keypoints)
print("Dirts in Selected area: ", len(dirt_keypoints))

print("*************************************")
#Display Number of detected Dotts
number_of_dot_blobs = len(dot_keypoints)
print("Dots in Selected area: ", len(dot_keypoints))
print("*************************************")

quantity_of_dirtDot = number_of_dot_blobs +number_of_dirt_blobs
print("Total Quantity of Dirt and Dots Detected In selected Area: ",quantity_of_dirtDot)
print("*************************************")

# Display cropped image
cv2.imshow("Dot Detection", dot_blobs)
cv2.imshow("Dirt Detection", dirt_blobs)

cv2.waitKey(0)
cv2.destroyAllWindows()