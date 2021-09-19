# VisionTrackVR by JanCraft
# Last Patch: 14/09/2021
#
#
#
#
#
#
# To calibrate, stay still in front of the camera, centered in
# your playspace and press 'C' while focusing the
# VisionTrackVR window.

print("[1/4] Importing libraries...")

import cv2
import mediapipe as mp
import socket
import time
import struct
import cfgloader
import math

print("[2/4] Loading configuration...")

cfg = cfgloader.load(open("config.cfg", 'r'))
calib = cfgloader.load(open("calibration.cfg", 'r'))

############
## CONFIG ##
############

CONFIG_SHOW_FPS = cfg.bool("show_fps")
CONFIG_CAMERA_INDEX = cfg.int("camera_index")

CONFIG_TRACKER_CONFIDENCE = cfg.float("tracker_confidence")

CONFIG_UDP_IP = cfg.str("udp_address")
CONFIG_UDP_PORT = cfg.int("udp_port")

############

TRACKER_LEFT_FOOT = [0, 0, 0]
TRACKER_RIGHT_FOOT = [0, 0, 0]
TRACKER_HIPS = [0, 0, 0]

fps_time = 0
fps_acc = 0
fps = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
lastkey = -1

def modx(x):
    return x * cfg.float("keypoint_scale")

def mody(y):
    return y * cfg.float("keypoint_scale")

def modz(z):
    return z * cfg.float("depth_keypoint_scale")

def vecsub(a, b):
    return {'x':a['x']-b['x'],'y':a['y']-b['y'],'z':a['z']-b['z']}

def veccross(a, b):
    return {'x':a['y'] * b['z'] - a['z'] * b['y'],'y':a['z'] * b['x'] - a['x'] * b['z'],'z':a['x'] * b['y'] - a['y'] * b['x']}

def vecdot(a, b):
    return a['x']*b['x'] + a['y']*b['y'] + a['z']*b['z']

def vecsqrmag(v):
    return v['x'] * v['x'] + v['y'] * v['y'] + v['z'] * v['z']

def clamp(x, _min, _max):
    return min(max(x, _min), _max)

def vecangle(a, b):
    denominator = math.sqrt(vecsqrmag(a) * vecsqrmag(b))
    if denominator < (10.0 ** -15): return 0.0

    dot = clamp(vecdot(a, b) / denominator, -1.0, 1.0)
    return math.degrees(math.acos(dot))

def vecnormalize(v):
    l = math.sqrt(vecsqrmag(v))
    return {'x':v['x']/l,'y':v['y']/l,'z':v['z']/l}

def vecscale(v, s):
    return {'x':v['x']*s,'y':v['y']*s,'z':v['z']*s}

def quatvector(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return {'x':roll_x,'y':pitch_y,'z':yaw_z}

def quatangleaxis(angle, axis):
    rad = math.radians(angle * 0.5)
    axis2 = vecscale(axis, math.sin(rad))
    return quatvector(axis2['x'], axis2['y'], axis2['z'], math.cos(rad))

def fromtorotation(aFrom, aTo):
    axis = veccross(aFrom, aTo)
    angle = vecangle(aFrom, aTo)
    return quatangleaxis(angle, vecnormalize(axis))

def angle_from_points(a, b):
    return fromtorotation({'x':0,'y':1,'z':0}, vecsub(b, a))

keypoints = []
def get_keypoints(pose_landmarks):
    keypoints.clear()
    if pose_landmarks is None: return keypoints
    if pose_landmarks.landmark is None: return keypoints
    for data_point in pose_landmarks.landmark:
        keypoints.append({ 'x': data_point.x, 'y': data_point.y, 'z': data_point.z, 'visibility': data_point.visibility })
    return keypoints

print("[3/4] Loading camera...")
cap = cv2.VideoCapture(CONFIG_CAMERA_INDEX)
print("[4/4] Setting up camera...")
cap.set(cv2.CAP_PROP_FPS, cfg.int("requested_fps"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

print("Starting...")

cv2.namedWindow("VisionTrackVR")

with mp_pose.Pose(
        smooth_landmarks=True,
        model_complexity=cfg.int('model_complexity'),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        tracker_count = 0
        results = pose.process(rgb)

        keypoints = get_keypoints(results.pose_landmarks)
        if len(keypoints) > 1:
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            hips = {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2,
                'z': (left_hip['z'] + right_hip['z']) / 2,
                'visibility': (left_hip['visibility'] + right_hip['visibility']) / 2
            }

            left_ankle = keypoints[27]
            right_ankle = keypoints[28]

            TRACKER_HIPS[0] = modx(hips['x'])
            TRACKER_HIPS[1] = mody(1 - hips['y'])
            TRACKER_HIPS[2] = modz(hips['z'])
            
            TRACKER_LEFT_FOOT[0] = modx(left_ankle['x'])
            TRACKER_LEFT_FOOT[1] = mody(1 - left_ankle['y'])
            TRACKER_LEFT_FOOT[2] = modz(left_ankle['z'])

            TRACKER_RIGHT_FOOT[0] = modx(right_ankle['x'])
            TRACKER_RIGHT_FOOT[1] = mody(1 - right_ankle['y'])
            TRACKER_RIGHT_FOOT[2] = modz(right_ankle['z'])

            hiprot = math.degrees(math.atan2(left_hip['x'] - right_hip['x'], right_hip['z'] - left_hip['z']))

            if lastkey == ord('c'):
                calib.values['hips_x'] = -TRACKER_HIPS[0] + 0
                calib.values['hips_y'] = -TRACKER_HIPS[1] + 0.5
                calib.values['hips_z'] = -TRACKER_HIPS[2] + 0
                calib.values['hips_r'] = -hiprot

                calib.values['left_foot_x'] = -TRACKER_LEFT_FOOT[0] + 0.25
                calib.values['left_foot_y'] = -TRACKER_LEFT_FOOT[1] + -1
                calib.values['left_foot_z'] = -TRACKER_LEFT_FOOT[2] + 0

                calib.values['right_foot_x'] = -TRACKER_RIGHT_FOOT[0] + -0.25
                calib.values['right_foot_y'] = -TRACKER_RIGHT_FOOT[1] + -1
                calib.values['right_foot_z'] = -TRACKER_RIGHT_FOOT[2] + 0

                cfgloader.dump(open('calibration.cfg', 'w'), calib, "Auto-generated calibration values")
            
            if lastkey == ord('r'):
                cfg = cfgloader.load(open("config.cfg", 'r'))
                calib = cfgloader.load(open("calibration.cfg", 'r'))

            if hips['visibility'] > CONFIG_TRACKER_CONFIDENCE: tracker_count += 1
            if left_ankle['visibility'] > CONFIG_TRACKER_CONFIDENCE: tracker_count += 1
            if right_ankle['visibility'] > CONFIG_TRACKER_CONFIDENCE: tracker_count += 1

            sock.sendto(struct.pack("dddddddddd", 
                TRACKER_HIPS[0] + calib.float("hips_x"), TRACKER_HIPS[1] + calib.float("hips_y"), TRACKER_HIPS[2] + calib.float("hips_z"),
                hiprot + 270 + calib.float("hips_r"),
                TRACKER_LEFT_FOOT[0] + calib.float("left_foot_x"), TRACKER_LEFT_FOOT[1] + calib.float("left_foot_y"), TRACKER_LEFT_FOOT[2] + calib.float("left_foot_z"),
                TRACKER_RIGHT_FOOT[0] + calib.float("right_foot_x"), TRACKER_RIGHT_FOOT[1] + calib.float("right_foot_y"), TRACKER_RIGHT_FOOT[2] + calib.float("right_foot_z")
            ), (CONFIG_UDP_IP, CONFIG_UDP_PORT))

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        fps_acc += 1
        if time.time() - fps_time >= 1:
            fps = fps_acc
            fps_acc = 0
            if CONFIG_SHOW_FPS: print(fps)
            fps_time = time.time()

        cv2.imshow('VisionTrackVR', image)
        cv2.setWindowTitle('VisionTrackVR', "VisionTrackVR - Trackers " + str(tracker_count) + "/3")
        lastkey = cv2.waitKey(1) & 0xFF
        if lastkey == 27:
            break

cv2.destroyAllWindows()
cap.release()
