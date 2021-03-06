from image_processing import *

class LaneDetector:
    """ Detect lane lines
    Attributes:
        transformer: perspective transformer
        calibration: calibration instance
    """

    def __init__(self, calibration, transformer):
        """ Inititalize lane detector
        Args:
            transformer: perspective transformer
            calibration: calibration instance
        """
        self.calibration = calibration
        self.transformer = transformer

    def process_image(self, img):
        """ Process each frame
        Args:
            img: camera image
        """
        objpoints, imgpoints = self.calibration.get_calibration_points()

        undistort_image = undistort(img, objpoints, imgpoints)
        processed_image = process_binary(undistort_image)
        processed_image = self.transformer.transform(processed_image)
        left_fit, right_fit, yvals, out_img = find_lanes(processed_image)
        processed_image = fit_lane(processed_image, undistort_image, yvals, left_fit, right_fit, self.transformer)
        left_curvature, right_curvature, distance = get_measurements(left_fit, right_fit, yvals)
        processed_image = draw_stat(processed_image, left_curvature, right_curvature, distance)

        return processed_image
