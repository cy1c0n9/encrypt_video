from classes import Lorenz
from classes import Protocol
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import math
import numpy as np
import time

# cap = cv2.VideoCapture(0)

video_folder = "./videos/"
filename = "bbt_test.mp4"
cap = cv2.VideoCapture(video_folder + filename)
fps = cap.get(cv2.CAP_PROP_FPS)

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

"""
    If the codec is lossless like ffv1, the encryption & decryption is just fine
    As lossless codec work successfully, the key synchronize should be correct 
"""
# video_writer = cv2.VideoWriter('./tmp/tmp_encrypted.avi', cv2.VideoWriter_fourcc('F', 'F', 'V', '1'), fps, size)
video_writer = cv2.VideoWriter('./tmp/tmp_encrypted.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

# the following line does not improve video quality as 1.0 is the maximamum parameter available
video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 1.0)

# Initializing parameters
master = Lorenz()
slave = Lorenz()

sender = Protocol(master)
receiver = Protocol(slave)

key = sender.get_sequence(25)
receiver.synchronize(key)

if not cap.isOpened():
    print("Error opening video stream or file")
    
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    start_time = time.time()
    # imgsplit = cv2.split(frame)
    """
        convert to Y_Cr_Cb format doesn't help
    """
    # cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb, frame)
    frame[10:100, 10:100] = sender.encrypt(frame[10:100, 10:100])
    # cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR, frame)
    # frame[10:100, 10:100] = receiver.decrypt(frame[10:100, 10:100])
    print("--- Encrypt %s seconds ---" % (time.time() - start_time))

    video_writer.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture

video_writer.release()
cap.release()
cv2.destroyAllWindows()

"""
    Read the encrypted file and decrypt it
"""
cap2 = cv2.VideoCapture('./tmp/tmp_encrypted.avi')
while cap2.isOpened():
    # Capture frame-by-frame
    ret, frame = cap2.read()
    if frame is None:
        break
    start_time = time.time()
    # encrypt_image = sender.encrypt(frame.copy())
    frame[10:100, 10:100] = receiver.decrypt(frame[10:100, 10:100])
    print("--- decrypt %s seconds ---" % (time.time() - start_time))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()

