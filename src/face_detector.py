import os
import argparse
from time import time
import numpy as np
import pandas as pd
import cv2
import cvlib as cv
# from cvlib.object_detection import draw_bbox
from face_recognition.api import face_encodings, compare_faces


def parse_args():
    """
    Input arguments parser.
    """
    parser = argparse.ArgumentParser(description="Capture and display live camera video on Jetson Nano")
    parser.add_argument("--usb", dest="use_usb",
                        help="use USB webcam",
                        action="store_true")
    parser.add_argument("--cpu_only", dest="cpu_only",
                        help="use CPU only",
                        action="store_true")
    parser.add_argument("--verbose", dest="verbose",
                        help="print detection outputs",
                        action="store_true")
    parser.add_argument("--tolerance", dest="tolerance", default=0.6, type=float,
                        help="How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.")
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

def furnish_image_confidence(frame, faces, confidences):
    """Draws bounding boxes & prints detection confidence.

    Args:
        frame (3D-Numpy Array): cv2 image
        faces (list): list of detected faces
        confidences (list): list of face detection confidence

    Returns:.
        3D-Numpy Array: cv2 image
    """
    # loop through detected faces
    for idx, face in enumerate(faces):
        
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        text = "{:.2f}%".format(confidences[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write confidence percentage on top of face rectangle
        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)
        
    return frame

############
def create_face_images(frame, faces, new_size=(120,180)):
    """Extracts the facial images from the frame.

    Args:
        frame (3D-Numpy Array): cv2 image
        faces (list): list of detected faces
        new_size (tuple, optional): Width & Height of face image. Defaults to (120,180).

    Returns:
        list: List of 3D-Numpy arrays cv2 image of faces
    """
    # loop through detected faces
    face_imgs = []
    for idx, face in enumerate(faces):
        
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # crop the faces
        face_img = frame[startY:endY,startX:endX]
        face_img = cv2.resize(face_img,new_size)
        face_imgs.append(face_img)
        
    return face_imgs

def init_known_faces(dir="../data/", filename="faces.pkl"):
    if filename not in os.listdir(dir):
        df = pd.DataFrame(columns=["ID","name","encoding","confidence"])
    else:
        df = pd.read_pickle(os.path.join(dir,filename))
    return df

def save_known_faces(df, dir="../data/", filename="faces.pkl"):
    df.to_pickle(os.path.join(dir,filename))

def update_known_faces(df, name, enc, confidence, dir="../data/", filename="faces.pkl"):
    ID = 0 if len(df)==0 else df["ID"].max()+1
    new_row = {"ID":ID,"name":name,"encoding":enc,"confidence":confidence}
    new_df = df.append(new_row, ignore_index=True)
    return new_df

def name_to_faces(faces):
    pass
###########

def live_inference(args):

    window_title = "i-see-you-with-jetson"
    if args.use_usb:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=6), cv2.CAP_GSTREAMER)
        window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        test_enc = np.array([])
        # INITIATE KNOWN ENCODINGS
        known_faces_df = init_known_faces()

        while True:

            # Reset elapsed time    
            frame_start_time = time()

            ret, frame = cap.read()

            # apply face detection
            faces, confidences = cv.detect_face(frame, threshold=0.5, enable_gpu = not args.cpu_only)
            presence = len(faces)>0

            if presence:
                encs = face_encodings(frame,faces) # List of arrays
                # if len(test_enc)==0:
                #     test_enc = encs[0]
                # print(compare_faces([test_enc],encs[0]),confidences)
                names = []
                for i, enc in enumerate(encs):
                    comparison = compare_faces(known_faces_df["encoding"].to_list(), enc)
                    if max(comparison):
                        names.append(known_faces_df["names"][])

                # Furnish image
                frame = furnish_image_confidence(frame, faces, confidences)


            # Display output
            cv2.imshow(window_title, frame)

            if args.verbose:
                print(f"Frame time: {time()-elapsed_time:.3f}s | Face: {faces} | Confidences: {confidences}")

            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    live_inference(args=args)

# windows: python -m src.face_detector --usb --cpu
# jetson: python -m src.face_detector