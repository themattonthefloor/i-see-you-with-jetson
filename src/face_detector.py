import argparse
import numpy as np
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from time import time


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Capture and display live camera video on Jetson TX2/TX1")
    parser.add_argument("--usb", dest="use_usb",
                        help="use USB webcam",
                        action="store_true")
    parser.add_argument("--cpu_only", dest="cpu_only",
                        help="use CPU only",
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

def live_inference(args):

    window_title = "i-see-you-with-jetson"
    if args.use_usb:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=6), cv2.CAP_GSTREAMER)
        window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        elapsed_time = time()
        while True:
            ret, frame = cap.read()

            # apply face detection
            face, confidence = cv.detect_face(frame, threshold=0.5, enable_gpu = not args.cpu_only)
            
            print(f"{time()-elapsed_time:.2f} S")
            elapsed_time = time()
            print(face)
            print(confidence)

            # loop through detected faces
            for idx, f in enumerate(face):
                
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                text = "{:.2f}%".format(confidence[idx] * 100)

                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write confidence percentage on top of face rectangle
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0,255,0), 2)

            # display output
            cv2.imshow(window_title, frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    live_inference(args=args)