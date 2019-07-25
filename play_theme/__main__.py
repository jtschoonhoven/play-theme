"""
With minor changes, this has been adapted from:
https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
"""
import cv2
import face_recognition
import numpy as np
import os
import pygame
import re
import time
from os import path

MODULE_DIRPATH = path.dirname(path.abspath(__file__))
PHOTOS_DIRPATH = path.join(MODULE_DIRPATH, '..', 'photos')
SOUNDS_DIRPATH = path.join(MODULE_DIRPATH, '..', 'sounds')

PHOTO_FILENAMES = [f for f in os.listdir(PHOTOS_DIRPATH) if f != '.gitignore']
SOUND_FILENAMES = [f for f in os.listdir(SOUNDS_DIRPATH) if f != '.gitignore']
PERSON_NAMES = [re.split('[^a-zA-Z]+', f)[0].lower() for f in PHOTO_FILENAMES]

# This is a demo of running face recognition on live video from your webcam.
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.


def _filename_prefix(filename):
    return re.split('[^a-zA-Z]+', filename)[0].lower()


# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []
for filename in PHOTO_FILENAMES:
    photo_path = path.join(PHOTOS_DIRPATH, filename)
    photo_subject = _filename_prefix(filename)
    photo = face_recognition.load_image_file(photo_path)
    encoded = face_recognition.face_encodings(photo)[0]
    known_face_encodings.append(encoded)
    known_face_names.append(photo_subject)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (for face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # if there was a match, play the theme song for the first matched face
    if face_names:
        name = face_names[0]
        # match name to song
        song = [f for f in SOUND_FILENAMES if _filename_prefix(f) == name][0]

        # play song
        pygame.mixer.init()
        pygame.mixer.music.load(path.join(SOUNDS_DIRPATH, song))
        pygame.mixer.music.play()
        time.sleep(15)
        pygame.mixer.music.fadeout(5000)
        face_names = []

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
