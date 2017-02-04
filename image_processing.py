import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibration import Calibration

def undistort(img, objpoints, imgpoints):
    """ Undistort image
    Args:
        img: image in BGR
        objpoints: correct image points
        imgpoints: corresponding distorted image points
    Returns:
        Undistorted image
    """
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst
