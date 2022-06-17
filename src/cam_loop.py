import numpy as np
import argparse
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Capture and display live camera video on Jetson TX2/TX1")
    parser.add_argument("--usb", dest="use_usb",
                        help="use USB webcam",
                        action="store_true")
    args = parser.parse_args()
    return args

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def show_camera(use_usb=False):
    window_title = "Camera"
    if use_usb:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, img = cap.read()
            cv2.imshow(window_title,img)
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    show_camera(use_usb=args.use_usb)