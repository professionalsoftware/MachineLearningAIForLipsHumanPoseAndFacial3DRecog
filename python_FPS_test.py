#!/usr/bin/python3
import lipsbodypose
import cv2
import numpy
import math
import sys
import time;      # This is required to include time module.

global rgb
global depth
global skeleton2d
global skeleton3d

pose = lipsbodypose.lipsbodypose()

t = 1
ticks_1 = time.time()
ticks_diff = 0.0
old_FPS = 0.0
counter = 0.0
while True:
    ( rgb, depth, skeleton2d, skeleton3d, humanID ) = pose.readFrame()

    counter = counter + 1.0;

    ticks_diff = time.time()-ticks_1
    #ticks_1 = time.time()

    #print("readFrame")
    FPS = counter/ticks_diff
    #FPS = (FPS + old_FPS) * 0.5;
    #old_FPS = FPS
    print("FPS:", FPS)
    #print("FPS:", 1.0//ticks_diff)
    #print("FPS:", 1.0/ticks_diff)

