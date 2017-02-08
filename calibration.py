import numpy as np
import cv2
import glob
import pickle
import os

class Calibration:
    """ Calibrate images using chessboard images

    Attributes:
        dir: directory path of calibration images
    """

    def __init__(self, dir, nx, ny):
        """ Initialize calibration class
        Args:
            dir: directory path of calibration images
            nx: number of points in x direction
            ny: number of points in y direction
        """

        self.dir = dir
        self.nx = nx
        self.ny = ny
        self.objpoints, self.imgpoints = None, None

    def get_calibration_points(self):
        """ Get calibration points
        Returns:
            (objpoints, imgpoints) tuple
        """
        if self.objpoints is None or self.imgpoints is None:
            self.objpoints, self.imgpoints = self.calibrate()

        return (self.objpoints, self.imgpoints)

    def calibrate(self):
        """ Calibrate images

        Returns:
            Tuple of object points and corresponding image points in two separate arrays
            example: (objpoints, imgpoints)
        """
        # If calibration data exists then return
        if os.path.isfile('calibration.p'):
            with open('calibration.p', 'rb') as data_file:
                data = pickle.load(data_file)
            print("Calibration exists and returned")
            return (data['objpoints'], data['imgpoints'])

        nx, ny = self.nx, self.ny
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        objpoints = []
        imgpoints = []

        # Make a list of calibration images
        images = glob.glob(self.dir + '/*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            print("Calibrating image ", idx)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

        cv2.destroyAllWindows()
        data = {
            'objpoints' : objpoints,
            'imgpoints' : imgpoints
        }
        # Save as a file
        with open('calibration.p', "wb") as data_file:
            pickle.dump(data, data_file)
        print("Calibration done and returned")

        self.objpoints, self.imgpoints = objpoints, imgpoints

        return (self.objpoints, self.imgpoints)
