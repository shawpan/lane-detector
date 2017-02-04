import cv2

class PerspectiveTransformer:
    """ Transform perspective of images

    Attributes:
        src: source points
        dst: destination points
        M: perspective  transform matrix
        M_inv: inverse perspective transform matrix
    """
    
    def __init__(self, src, dst):
        """ Initialize perspective transformer class
        Args:
            src: source points
            dst: destination points
        """
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """ Transform perspective of image
        Args:
            img: input image
        """
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        """ Reverse perspective transform of image
        Args:
            img: input image
        """
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
