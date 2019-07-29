"""
With minor changes, this has been adapted from:
https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
"""
import cv2
import face_recognition
import numpy as np
import os
import pygame
import random
import re
import threading
import time
from cachetools import TTLCache
from os import path

MODULE_DIRPATH = path.dirname(path.abspath(__file__))
PHOTOS_DIRPATH = path.join(MODULE_DIRPATH, '..', 'photos')
SOUNDS_DIRPATH = path.join(MODULE_DIRPATH, '..', 'sounds')

PHOTO_FILENAMES = [f for f in os.listdir(PHOTOS_DIRPATH) if f != '.gitignore']
SOUND_FILENAMES = [f for f in os.listdir(SOUNDS_DIRPATH) if f != '.gitignore']
PERSON_NAMES = [re.split('[^a-zA-Z]+', f)[0].lower() for f in PHOTO_FILENAMES]

TTL_CACHE_SECONDS = 15
TTL_CACHE = TTLCache(maxsize=float('inf'), ttl=TTL_CACHE_SECONDS)

# This is a demo of running face recognition on live video from your webcam.
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.


def get_username_from_filename(filename: str) -> str:
    """Parse a filename and return the initial alphabetic characters.

    This is a convention of the `filename_prefix` library: users are only defined by the filename
    of their headshots and theme songs. To support a single user having multiple headshots and
    themes, we consider the initial alphabetic-part of the filename to be a username, and any
    additional characters to be an ID of the file.

    E.g. "jonathan.png", "jonathan_2.jpg", and "jonathan123.png" will all be attributed to the user
    "jonathan", but "jon.jpg", "jonathandoe.png", "jon-athan.png" would all be three separate
    users.
    """
    return re.split('[^a-zA-Z]+', filename)[0].lower()


def play_theme_for_username_in_thread(username: str) -> None:
    """Play theme song for named user in a new thread. Throws an error if no theme is defined."""
    # do nothing if a thread is currently alive
    if threading.active_count() > 1:
        return None
    # if user's theme has already been played within TTL, do nothing
    if TTL_CACHE.get(username):
        return None
    # add user to TTL cache to avoid playing too frequently
    TTL_CACHE[username] = True
    theme_filenames = [f for f in SOUND_FILENAMES if get_username_from_filename(f) == username]
    if theme_filenames:
        filename = random.choice(theme_filenames)
        thread = threading.Thread(target=_play_theme, args=[filename])
        thread.start()
    else:
        raise Exception(f'Failed to find theme for user {username}.')


def _play_theme(filename: str) -> None:
    """Play the theme song for a given user. Throws an error if no theme is defined."""
    pygame.mixer.music.load(path.join(SOUNDS_DIRPATH, filename))
    pygame.mixer.music.play()
    time.sleep(3)  # 15
    pygame.mixer.music.fadeout(5000)


def get_known_face_encodings():
    known_face_encodings = []
    for filename in sorted(PHOTO_FILENAMES):
        photo_path = path.join(PHOTOS_DIRPATH, filename)
        photo = face_recognition.load_image_file(photo_path)
        encoded = face_recognition.face_encodings(photo)[0]
        known_face_encodings.append(encoded)
    return known_face_encodings


def get_known_face_names():
    """"""
    known_face_names = []
    for filename in sorted(PHOTO_FILENAMES):
        photo_subject = get_username_from_filename(filename)
        known_face_names.append(photo_subject)
    return known_face_names


def process_frame(frame, known_face_encodings, known_face_names) -> None:
    """"""
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (for face_recognition)
    rgb_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    matched_face_locations = face_recognition.face_locations(rgb_frame)
    matched_face_encodings = face_recognition.face_encodings(rgb_frame, matched_face_locations)
    matched_face_names = []

    for matched_face_encoding in matched_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, matched_face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, matched_face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            matched_face_names.append(name)

    # if there was a match, play the theme song for the first matched face
    if matched_face_names:
        username = random.choice(matched_face_names)
        play_theme_for_username_in_thread(username)

    return matched_face_locations, matched_face_names


def run(video_capture) -> None:
    """"""
    pygame.mixer.init()

    matched_face_locations = []
    matched_face_names = []

    # create arrays of known face encodings and their names
    known_face_encodings = get_known_face_encodings()
    known_face_names = get_known_face_names()

    skip_this_frame = False

    while True:
        # capture a single frame
        _, frame = video_capture.read()
        if not skip_this_frame:
            matched_face_locations, matched_face_names = process_frame(
                frame,
                known_face_encodings,
                known_face_names,
            )

        # process every other frame to save time
        skip_this_frame = not skip_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(matched_face_locations, matched_face_names):
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

        # display video feed with overlay
        cv2.imshow('Video', frame)


try:
    # get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    run(video_capture)
except (KeyboardInterrupt, SystemExit):
    video_capture.release()
    cv2.destroyAllWindows()
