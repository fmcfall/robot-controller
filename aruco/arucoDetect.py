import numpy as np
import cv2 as cv
import sys
import time
from readCamParams import *
from DLT import DLT

ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

def get_centre_points(frame, aruco_dict_type, matrix_coefficients, distortion_coefficient):
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.aruco_dict = cv.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters)

    cX, cY = 0, 0

    if len(corners) > 0:

        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            ret = True

    else:

        ret = False

    return cX, cY, ret

# def aruco_display(corners, ids, rejected, image):
# 	if len(corners) > 0:
# 		# flatten the ArUco IDs list
# 		ids = ids.flatten()
# 		# loop over the detected ArUCo corners
# 		for (markerCorner, markerID) in zip(corners, ids):
# 			# extract the marker corners (which are always returned in
# 			# top-left, top-right, bottom-right, and bottom-left order)
# 			corners = markerCorner.reshape((4, 2))
# 			(topLeft, topRight, bottomRight, bottomLeft) = corners
# 			# convert each of the (x, y)-coordinate pairs to integers
# 			topRight = (int(topRight[0]), int(topRight[1]))
# 			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
# 			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
# 			topLeft = (int(topLeft[0]), int(topLeft[1]))

# 			cv.line(image, topLeft, topRight, (0, 255, 0), 2)
# 			cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
# 			cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
# 			cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
# 			# compute and draw the center (x, y)-coordinates of the ArUco
# 			# marker
# 			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
# 			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
# 			cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
# 			# draw the ArUco marker ID on the image
# 			#cv.putText(image, (topLeft[0], topLeft[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
# 			#	0.5, (0, 255, 0), 2)
# 			#print("[Inference] ArUco marker ID: {}".format(markerID))
# 			# show the output image
# 	return image

# def detect_centre():

#     intr0, dist0 = read_camera_parameters(0)
#     intr1, dist1 = read_camera_parameters(1)

#     P0 = get_projection_matrix(0)
#     P1 = get_projection_matrix(1)

#     aruco_type = "DICT_5X5_250"

#     arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

#     arucoParams = cv.aruco.DetectorParameters_create()

#     video = cv.VideoCapture(1)

#     kpts_camL = []
#     kpts_camR = []
#     kpts_3d = []

#     while True:
#         success, img = video.read()
#         if not success: break

#         img0 = img[:, 0:img.shape[1]//2]
#         img1 = img[:, img.shape[1]//2:img.shape[1]]

#         image_height, image_width, _ = img.shape

#         x0, y0, ret0 = get_centre_points(img0, ARUCO_DICT[aruco_type], intr0, dist0)
#         x1, y1, ret1 = get_centre_points(img1, ARUCO_DICT[aruco_type], intr1, dist1)

#         point0 = [[x0, y0]]
#         point1 = [[x1, y1]]

#         for uv0, uv1 in zip(point0, point1):
#             if uv0[0] == -1 or uv1[0] == -1:
#                 _p3d = [-1, -1, -1]
#             else:
#                 _p3d = DLT(P0, P1, uv0, uv1) #calculate 3d position of keypoint

#         cv.imshow('cam0', img0)
#         cv.imshow('cam1', img1)

#         key = cv.waitKey(1)
#         # If q entered whole process will stop
#         if key == ord('q'):
#             break
        
#     cv.destroyAllWindows()

# if __name__ == "__main__":
     
#     intr0, dist0 = read_camera_parameters(0)
#     intr1, dist1 = read_camera_parameters(1)

#     P0 = get_projection_matrix(0)
#     P1 = get_projection_matrix(1)

#     aruco_type = "DICT_5X5_250"
#     arucoDict = ARUCO_DICT[aruco_type]
#     cv.aruco_dict = cv.aruco.Dictionary_get(arucoDict)
#     parameters = cv.aruco.DetectorParameters_create()

#     video = cv.VideoCapture(0)
#     while True:
#         success, img = video.read()
#         if not success: break

#         img0 = img
#         gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
#         corners0, ids0, rejected_img_points0 = cv.aruco.detectMarkers(gray0, cv.aruco_dict, parameters=parameters)
#         #print(ids0)
#         detected_markers0 = aruco_display(corners0, ids0, rejected_img_points0, img0)

#         rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners0, 0.02, intr0,
#                                                                        dist0)

#         # Draw a square around the markers
#         cv.aruco.drawDetectedMarkers(img0, corners0) 

#         # Draw Axis
#         cv.aruco.drawAxis(img0, intr0, dist0, rvec, tvec, 0.01)  

#         cv.imshow("ArUco Detection", img0)


#         key = cv.waitKey(1) & 0xFF
#         if key == ord("q"):
# 	        break
#         if key == ord("s"):
#             cv.imwrite("Arcuo.png", img0)


# def aruco_display(corners, ids, rejected, image):
    
#     if len(corners) > 0:
        
#         ids = ids.flatten()
        
#         for (markerCorner, markerID) in zip(corners, ids):
            
#             corners = markerCorner.reshape((4, 2))
#             (topLeft, topRight, bottomRight, bottomLeft) = corners
            
#             topRight = (int(topRight[0]), int(topRight[1]))
#             bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#             bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#             topLeft = (int(topLeft[0]), int(topLeft[1]))
            
#             cX = int((topLeft[0] + bottomRight[0]) / 2.0)
#             cY = int((topLeft[1] + bottomRight[1]) / 2.0)
#             #print("[Inference] ArUco marker ID: {}".format(markerID))
            
#     return cX, cY

# def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.aruco_dict = cv.aruco.Dictionary_get(aruco_dict_type)
#     parameters = cv.aruco.DetectorParameters_create()

#     x, y = 0, 0

#     corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters)

#     if len(corners) > 0:
#         for i in range(0, len(ids)):
#             rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                       distortion_coefficients)
            
#             x, y = aruco_display(corners, ids, rejected_img_points, frame)
            
#             ret = True

#             cv.aruco.drawDetectedMarkers(frame, corners) 
#             cv.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
            
#             #cv.putText(frame, str(markerPoints), [100, 100], cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

#     else:
#             ret = False

#     return frame, x, y, ret

# intr0, dist0 = read_camera_parameters(0)
# intr1, dist1 = read_camera_parameters(1)

# P0 = get_projection_matrix(0)
# P1 = get_projection_matrix(1)

# aruco_type = "DICT_5X5_250"

# arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

# arucoParams = cv.aruco.DetectorParameters_create()

# video = cv.VideoCapture(1)

# kpts_camL = []
# kpts_camR = []
# kpts_3d = []

# while True:
#     success, img = video.read()
#     if not success: break

#     img0 = img[:, 0:img.shape[1]//2]
#     img1 = img[:, img.shape[1]//2:img.shape[1]]

#     image_height, image_width, _ = img.shape

#     output0, x0, y0, ret0 = pose_estimation(img0, ARUCO_DICT[aruco_type], intr0, dist0)
#     output1, x1, y1, ret1 = pose_estimation(img1, ARUCO_DICT[aruco_type], intr1, dist1)

#     point0 = [[x0, y0]]
#     point1 = [[x1, y1]]

#     for uv0, uv1 in zip(point0, point1):
#         if uv0[0] == -1 or uv1[0] == -1:
#             _p3d = [-1, -1, -1]
#         else:
#             _p3d = DLT(P0, P1, uv0, uv1) #calculate 3d position of keypoint

#     cv.imshow('cam0', img0)
#     cv.imshow('cam1', img1)

#     key = cv.waitKey(1)
#     # If q entered whole process will stop
#     if key == ord('q'):
#         break
#     if key == ord('s'):
#         cv.imwrite("arucoL.png", img0)
#         cv.imwrite("arucoR.png", img1)
    
# cv.destroyAllWindows()