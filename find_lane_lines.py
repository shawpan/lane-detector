import numpy as np
import cv2
import glob
import pickle
import os
import ntpath
import matplotlib.pyplot as plt
from image_processing import *
from perspective_transformer import PerspectiveTransformer
from moviepy.editor import VideoFileClip
import argparse

calibration = Calibration('camera_cal', 9, 5)
objpoints, imgpoints = calibration.calibrate()

def process_image(img):
    """ Process each frame
    Args:
        img: camera image
    """
    src = np.float32([[240,719],[579,450],[712,450],[1165,719]])
    dst =  np.float32([[300,719],[300,0],[900,0],[900,719]])
    transformer = PerspectiveTransformer(src, dst)

    undistort_image = undistort(img, objpoints, imgpoints)
    processed_image = process_binary(undistort_image)
    processed_image = transformer.transform(processed_image)
    left_fit, right_fit, yvals, out_img = find_lanes(processed_image)
    processed_image = fit_lane(processed_image, undistort_image, yvals, left_fit, right_fit, transformer)
    left_curvature, right_curvature, distance = get_curvature(left_fit, right_fit, yvals)
    processed_image = draw_stat(processed_image, left_curvature, right_curvature, distance)

    return processed_image

def find_lane_lines(type):
    """ Find lanes from images or video
    Args:
        type: type of action, image or video
    """
    if type == 'v':
        clip = VideoFileClip("./project_video.mp4")
        output_video = "./output_video/project_video.mp4"
        output_clip = clip.fl_image(process_image)
        output_clip.write_videofile(output_video, audio=False)
    elif type == 'i':
        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
            print('Processing image ', idx)
            image = cv2.imread(fname)
            processed_image = process_image(image)
            print('Processing done!!! ', idx)
            output_filename = 'output_images/' + ntpath.basename(fname)
            cv2.imwrite(output_filename, processed_image)
    else:
        print('Invalid type requested')

def doc():
    """ Execute and generate debug images
    """
    src = np.float32([[240,719],[579,450],[712,450],[1165,719]])
    dst =  np.float32([[300,719],[300,0],[900,0],[900,719]])
    transformer = PerspectiveTransformer(src, dst)
    image = cv2.imread('test_images/test4.jpg')
    cv2.imwrite('doc/distorted.jpg', image)
    image = cv2.imread('doc/distorted.jpg')
    undistort_image = undistort(image, objpoints, imgpoints)
    cv2.imwrite('doc/undistorted.jpg', undistort_image)
    processed_image = process_binary(undistort_image)
    cv2.imwrite('doc/binary.jpg', processed_image*255)
    processed_image = transformer.transform(processed_image)
    cv2.imwrite('doc/birdeyeview.jpg', processed_image*255)
    left_fit, right_fit, yvals, out_img = find_lanes(processed_image)
    cv2.imwrite('doc/lanes.jpg', out_img)
    processed_image = fit_lane(processed_image, undistort_image, yvals, left_fit, right_fit, transformer)
    left_curvature, right_curvature, distance = get_curvature(left_fit, right_fit, yvals)
    processed_image = draw_stat(processed_image, left_curvature, right_curvature, distance)
    cv2.imwrite('doc/final.jpg', out_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='type i/v/d')
    args = parser.parse_args()
    if args.type == 'd':
        doc()
    else:
        find_lane_lines(args.type)
