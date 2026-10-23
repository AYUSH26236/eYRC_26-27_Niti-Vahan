'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          NV_6236
# Author List:      Utkarsh Singh, Ayush Tiwari, Anushka Telore, Atharva Jadhav
# Filename:         task2a.py
# Functions:        detect_lane, compute_control
# Global variables: WHEELBASE, TRACK, MAX_STEER, WHEEL_RADIUS, CRUISE, LANE_WIDTH_M, ROI_TOP,
#                   SCAN_STEP, NEAR_ROW, AXIS_X, YELLOW_LO, YELLOW_HI, WHITE_LO, WHITE_HI,
#                   K_STANLEY, K_SOFT, PSI_GAIN, SMOOTH, MAX_MISS, _S


####################### IMPORT MODULES #######################
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

import math

# ---------------- TUNE: vehicle (run probe_scene.py to read these) ----------------
WHEELBASE = 0.30                 # m, rear axle to front steering axis
TRACK = 0.20                     # m, distance between the two steering joints
MAX_STEER = math.radians(40.0)   # stay inside the 45 deg limit
WHEEL_RADIUS = 0.05              # m
CRUISE = 4.0                     # rad/s on both drive motors

# ---------------- TUNE: lane / camera ----------------
LANE_WIDTH_M = 0.40              # real width of one lane in metres (sets the e scale)
ROI_TOP = 0.45                   # ignore the top part of the frame (fraction of height)
SCAN_STEP = 4                    # px between scanned rows
NEAR_ROW = 0.90                  # row (fraction of height) where e and psi are read
AXIS_X = None                    # vehicle axis column in px; None = image centre
YELLOW_LO, YELLOW_HI = (15, 80, 80), (40, 255, 255)
WHITE_LO, WHITE_HI = (0, 0, 170), (180, 60, 255)

# ---------------- TUNE: controller ----------------
K_STANLEY = 1.5                  # cross-track gain
K_SOFT = 0.05                    # keeps atan() sane at very low speed
PSI_GAIN = 0.5                   # perspective view exaggerates slope; scale it down
SMOOTH = 0.5                     # 0..1, higher = smoother but slower to react
MAX_MISS = 8                     # frames to hold the last estimate when detection fails

# state kept between frames
_S = {'rw': {}, 'rc': {}, 'last': None, 'miss': 0, 'delta': 0.0}


def _marking_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.bitwise_or(cv2.inRange(hsv, YELLOW_LO, YELLOW_HI),
                       cv2.inRange(hsv, WHITE_LO, WHITE_HI))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _ackermann(delta):
    t = math.tan(delta)
    left = math.atan2(WHEELBASE * t, WHEELBASE - 0.5 * TRACK * t)
    right = math.atan2(WHEELBASE * t, WHEELBASE + 0.5 * TRACK * t)
    return left, right


def detect_lane(frame):
    '''
    Purpose:
    ---
    Work out, from a single camera frame, how wrong the vehicle currently is -
    and hand back whatever `compute_control()` needs in order to correct it.

    This is your Task 1C detector, extended. Task 1C answered *where the lane
    centre is* and *which lane the vehicle is in*; a controller cannot use
    either of those directly. It needs to know how far the vehicle is from
    where it should be, and which way it is pointing.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        One BGR frame from the vehicle's camera, of shape (height, width, 3),
        the right way up and in the right colour order.

    Returns:
    ---
    Anything you like - a tuple, a dictionary, your own class. Whatever you
    return here is passed straight to compute_control() and nowhere else, so
    the two functions only have to agree with each other.

    Example call:
    ---
    detection = detect_lane(frame)

    REMEMBER:
    ---
    Both lanes are driven during evaluation, and the arena looks mirrored from
    the other one. Do not hard-code which side the boundary is on.

    Do NOT call cv2.imshow() or cv2.imwrite() from inside this function.

    You are marked on how closely you hold the MIDDLE of your lane, not merely
    on staying the correct side of the dashed line.
    '''

    # Returns (e, psi) with e in metres and psi in radians, both positive to the
    # LEFT of the vehicle, or None when no usable lane was found.
    h, w = frame.shape[:2]
    cx = AXIS_X if AXIS_X is not None else (w - 1) / 2.0
    mask = _marking_mask(frame)
    y0 = int(h * ROI_TOP)

    rows, cents = [], []
    for y in range(h - 1, y0, -SCAN_STEP):
        cols = np.flatnonzero(mask[y])
        if cols.size == 0:
            continue
        left = cols[cols < cx - 3]
        right = cols[cols > cx + 3]
        l = int(left[-1]) if left.size else None     # nearest marking on the left
        r = int(right[0]) if right.size else None    # nearest marking on the right
        wk = _S['rw'].get(y)
        prev = _S['rc'].get(y, cx)
        c = None
        if l is not None and r is not None:
            g = r - l
            if wk is None:
                c = 0.5 * (l + r)
                _S['rw'][y] = float(g)
            elif g < 0.6 * wk:
                continue                              # stray blob, skip this row
            elif g <= 1.35 * wk:
                c = 0.5 * (l + r)
                _S['rw'][y] = 0.9 * wk + 0.1 * g
            else:
                # gap too wide: a dash of the divider is missing, so trust the
                # marking that keeps the centre closest to where it was
                c1, c2 = l + 0.5 * wk, r - 0.5 * wk
                c = c1 if abs(c1 - prev) <= abs(c2 - prev) else c2
        elif wk is not None and l is not None:
            c = l + 0.5 * wk
        elif wk is not None and r is not None:
            c = r - 0.5 * wk
        if c is None:
            continue
        _S['rc'][y] = c
        rows.append(y)
        cents.append(c)

    if len(rows) < 5:
        _S['miss'] += 1
        return None

    ys = np.array(rows, dtype=float)
    xs = np.array(cents, dtype=float)
    deg = 2 if len(ys) >= 8 else 1
    co = np.polyfit(ys, xs, deg)
    res = np.abs(np.polyval(co, ys) - xs)
    keep = res <= max(3.0, 2.5 * float(np.median(res)))
    if 5 <= keep.sum() < len(ys):
        co = np.polyfit(ys[keep], xs[keep], deg)

    yn = min(h * NEAR_ROW, float(ys.max()))
    xn = float(np.polyval(co, yn))
    dxdy = float(np.polyval(np.polyder(co), yn))

    near = [v for k, v in _S['rw'].items() if abs(k - yn) <= 16]
    if not near:
        _S['miss'] += 1
        return None
    m_per_px = LANE_WIDTH_M / float(np.median(near))

    e = (cx - xn) * m_per_px          # + when the lane centre is to the left
    psi = PSI_GAIN * math.atan(dxdy)  # + when the lane runs off to the left
    _S['miss'] = 0
    _S['last'] = (e, psi)
    return (e, psi)


def compute_control(detection, stop_requested):
    '''
    Purpose:
    ---
    Turn the errors from detect_lane() into the three numbers the vehicle is
    driven with, and obey the stop flag.

    This is where your Task 1B controller and your Task 1A steering geometry
    meet: work out one steering angle from the error, then split it into the
    two front-wheel angles the joints actually take.

    Input Arguments:
    ---
    `detection` :   [ any ]
        Whatever detect_lane() returned for this frame.

    `stop_requested` :  [ bool ]
        True while the evaluation is holding the vehicle at a stop point.
        This is a LEVEL, not a pulse: it stays True for the whole duration of
        the stop and goes False when the wait is over. While it is True the
        vehicle must be STATIONARY, so return a speed of 0.0. There is no
        stopwatch for you to write and no duration to hard-code.

    Returns:
    ---
    `left_angle`, `right_angle` :   [ float ]
        The two front-wheel steering angles, in RADIANS, written straight to
        the steering joints. The scene's steering joints are POSITIVE TO THE
        LEFT. Nothing is clamped for you - the vehicle's own limit is 45
        degrees, and commanding past it will not steer any harder.

    `speed` :   [ float ]
        Target velocity for both drive motors, in rad/s. 0.0 holds the
        vehicle still.

    Example call:
    ---
    left_angle, right_angle, speed = compute_control(detection, stop_requested)

    REMEMBER:
    ---
    Keep the loop running while stopped - set the SPEED to zero, not your
    program. If you stop reading frames you will not see the flag drop.

    Sooner or later a frame comes back with nothing usable in it. Holding the
    last steering angle, easing off the throttle and stopping dead are all
    defensible - doing nothing is not.
    '''

    speed = CRUISE
    if detection is None:
        # hold the last good estimate briefly, then straighten up and creep
        if _S['last'] is not None and _S['miss'] <= MAX_MISS:
            e, psi = _S['last']
        else:
            e, psi = 0.0, 0.0
            speed = 0.3 * CRUISE
    else:
        e, psi = detection

    v = max(CRUISE * WHEEL_RADIUS, 0.05)
    delta = psi + math.atan2(K_STANLEY * e, v + K_SOFT)       # Stanley
    delta = max(-MAX_STEER, min(MAX_STEER, delta))
    delta = SMOOTH * _S['delta'] + (1.0 - SMOOTH) * delta
    _S['delta'] = delta

    left_angle, right_angle = _ackermann(delta)               # Task 1A split
    if stop_requested:
        return left_angle, right_angle, 0.0
    return left_angle, right_angle, speed


##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())