import os
import argparse
from time import time
import numpy as np
import pandas as pd
import cv2
import cvlib as cv
# from cvlib.object_detection import draw_bbox
from face_recognition.api import face_locations, face_encodings, compare_faces, face_distance


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
    parser.add_argument("--match_tolerance", dest="tolerance", default=0.6, type=float,
                        help="How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.")
    parser.add_argument("--img_dir", dest="img_dir", default="data", type=str,
                        help="directory to save facial images")
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

def furnish_image_confidence(frame, faces, confidences, names):
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

        # Draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # Print name and confidence
        text = f"{names[idx]} ({(confidences[idx] * 100):.2f})" #"{:.2f}%".format(confidences[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
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
    # Loop through detected faces
    face_imgs = []
    for idx, face in enumerate(faces):
        
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # Crop the faces
        face_img = frame[startY:endY,startX:endX]
        face_img = cv2.resize(face_img,new_size)
        face_imgs.append(face_img)
        
    return face_imgs

def init_known_faces(dir="data", pkl_filename="faces.pkl"):
    """Initializes known faces from the pickle file,
    updates the names inferred from image filenames,
    and returns the DataFrame

    Args:
        dir (str, optional): Directory of facial images and pickle file. Defaults to "data".
        pkl_filename (str, optional): Filename of pickle file. Defaults to "faces.pkl".

    Returns:
        Pandas DataFrame: Dataframe of names, encoding & confidence
    """
    if pkl_filename not in os.listdir(dir):
        df = pd.DataFrame(columns=["name","encoding","confidence"])
    else:
        df = pd.read_pickle(os.path.join(dir,pkl_filename))
        names_dict = [x.replace(".jpg","").split("_") for x in os.listdir(dir) if x.endswith(".jpg")]
        names_dict = {int(x[0]): (x[1] if len(x)>1 else "Unknown") for x in names_dict}
        df["name"] = df.index.to_series().map(names_dict)
    return df

def save_known_faces(df, dir="data", filename="faces.pkl"):
    """Saves the face encodings dataframe to a pickle file.

    Args:
        df (Pandas DataFrame): Dataframe of names, encoding & confidence
        dir (str, optional): Directory to save. Defaults to "data".
        filename (str, optional): Filename to save. Defaults to "faces.pkl".
    """
    df.to_pickle(os.path.join(dir,filename))

def update_known_faces(df, name, enc, confidence, dir="data", filename="faces.pkl"):
    """Updates the known face encodings dataframe.

    Args:
        df (Pandas DataFrame): Dataframe of names, encoding & confidence
        name (str): Name
        enc (np.array): Encoding
        confidence (float): Confidence
        dir (str, optional): _description_. Defaults to "data".
        filename (str, optional): _description_. Defaults to "faces.pkl".

    Returns:
        Pandas DataFrame: Dataframe of names, encoding & confidence
    """
    new_row = {"name":name,"encoding":enc,"confidence":confidence}
    new_df = df.append(new_row, ignore_index=True)
    return new_df

def get_details(df, idx):
    details = df.iloc[idx]
    if details['name']=="Unknown":
        name = "#" + str(idx)
    else:
        name = details['name']
    return name, details['encoding'], details['confidence']

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
        # INITIATE KNOWN ENCODINGS
        known_faces_df = init_known_faces()
        print(known_faces_df)

        while True:

            # Reset elapsed time    
            frame_start_time = time()

            ret, frame = cap.read()

            # apply face detection
            faces, confidences = cv.detect_face(frame, threshold=0.5, enable_gpu = not args.cpu_only)
            faces_2 = face_locations(frame, model='hog')
            print(faces, faces_2)
            presence = len(faces)>0

            if presence:
                encs = face_encodings(frame,faces)
                names = []
                for i, enc in enumerate(encs):
                    comparison = face_distance(known_faces_df["encoding"].tolist(), enc)
                    print(comparison)
                    if len(comparison)>0 and min(comparison)<args.tolerance:
                        name, _, _ = get_details(known_faces_df, np.argmin(comparison))
                        names.append(name)
                    else: # Update name, encoding, confidence, facial image file
                        name = "Unknown"
                        names.append(name)
                        confidence = confidences[i]
                        known_faces_df = update_known_faces(known_faces_df, name, enc, confidence)

                        ID_str = str(known_faces_df.index.max()).zfill(4)
                        face_img = create_face_images(frame, [faces[i]])[0]
                        face_img_save_path = os.path.join(args.img_dir,ID_str)+".jpg"
                        cv2.imwrite(face_img_save_path, face_img)


                # Furnish image
                frame = furnish_image_confidence(frame, faces, confidences, names)

            # Display output
            cv2.imshow(window_title, frame)

            if args.verbose:
                print(f"Frame time: {time()-frame_start_time:.3f}s | Face: {faces} | Confidences: {confidences}")


            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_known_faces(known_faces_df)

if __name__ == '__main__':
    args = parse_args()
    live_inference(args=args)

# windows: python -m src.face_detector --usb --cpu
# jetson: python -m src.face_detector