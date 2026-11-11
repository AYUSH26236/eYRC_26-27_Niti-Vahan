'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2C of Niti Vahan (NV) Theme.
*
*  Filename:         task2c.py
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

# Team ID:          < NV_6236 >
# Author List:      < Ayush Tiwari, Anushka Telore, Atharva Jadhav, Utkarsh Singh >
# Filename:         task2c.py
# Functions:        parking_manoeuvre, _path_steering, _speed_for, _marker_lost
# Global variables: CRUISE_SPEED, CREEP_SPEED, CREEP_FROM, CRUISE_UNTIL, PLAN_END_FORWARD,
#                   MIN_PLAN_RANGE, STOP_LEAD, MAX_STEERING, HEADING_CAP, BEARING_LIMIT,
#                   PATH_GAIN, STEER_GAIN, HOLD_FRAMES, GIVE_UP_FRAMES, SEARCH_STEERING,
#                   _last_steering, _lost_frames, _last_bearing


####################### IMPORT MODULES #######################
import argparse
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2C.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# Two parking bays sit side by side to the RIGHT of the road, long axis along
# it. Each carries an ArUco marker at its head, facing back down the road:
# id 0 on bay 0 (nearer the road), id 1 on bay 1 (further right). The vehicle
# is spawned on the road, parallel to the bays, short of them and to their left.
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Which bay to park in: 0 or 1. This is the default - `python task2c.py --bay 1`
# overrides it for one run without editing the file.
TARGET_ID = 1

# Bay geometry, for your parking logic. Distances are measured along the road
# from the vehicle's centre to the line the markers stand on.
BAY_MOUTH_FORWARD  = 0.37    # m - the vehicle's centre is at the bay mouth here
BAY_MIDDLE_FORWARD = 0.19    # m - ...and in the middle of the bay here
BAY_WIDTH          = 0.22    # m - the vehicle is 0.18 m across
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

# ---- tuning constants ---------------------------------------------------------
CRUISE_SPEED     = 2.0                      # rad/s - kept low: the camera pose estimate drifts if the vehicle moves too far per frame
CREEP_SPEED      = 1.5                      # rad/s - the last stretch into the bay
CREEP_FROM       = 0.30                     # m - 'forward' at which we are fully down to creep speed
CRUISE_UNTIL     = 0.70                     # m - 'forward' above which we drive at full speed

PLAN_END_FORWARD = 0.55                     # m - 'forward' by which we want to be on the line and straight (mouth is 0.37)
MIN_PLAN_RANGE   = 0.10                     # m - floor on the distance left to plan over (keeps late steering gentle)
STOP_LEAD        = 0.004                    # m - stop this much early, so the roll-out ends on the mark

MAX_STEERING     = np.deg2rad(45.0)         # vehicle limit
HEADING_CAP      = np.deg2rad(42.0)         # steepest crossing: more and the marker leaves the camera view
BEARING_LIMIT    = np.deg2rad(43.0)         # marker this far off-centre in the image: stop turning further
PATH_GAIN        = 2.0                      # smooth crossing is about this much steeper than a straight line
STEER_GAIN       = 0.8                      # steering (rad) per rad of heading error (low: the pose is noisy)
HOLD_FRAMES      = 6                        # frames to keep steering blind after the marker is lost
GIVE_UP_FRAMES   = 80                       # frames without the marker before the vehicle stops
SEARCH_STEERING  = np.deg2rad(20.0)         # how hard to turn towards the marker's last side while it is lost

_last_steering = 0.0                        # steering sent on the previous frame
_lost_frames   = 0                          # consecutive frames without the marker
_last_bearing  = 0.0                        # where the marker was last seen in the camera view


def parking_manoeuvre(pose):
    '''
    Purpose:
    ---
    Park in the target bay by following a smooth path onto the bay's centre line.

    Idea:
    ---
    Do not steer at the marker. Instead, every frame, work out which HEADING
    would bring the vehicle onto the centre line in the distance that is left,
    and steer to hold that heading:

        wanted heading = -atan( PATH_GAIN * lateral / distance_left )

      * far from the line it asks for a steep crossing (capped, so the marker
        stays in the camera view) - this is the early, hard turn;
      * as lateral shrinks the wanted heading shrinks with it, so the vehicle
        straightens out by itself, before the bay mouth;
      * bay 1 (twice the offset) simply gets a steeper, longer turn than bay 0.

    Then it slows to a creep and stops in the middle of the bay.

    Input Arguments:
    ---
    `pose` :    [ dict or None ]   see the task description

    Returns:
    ---
    `steering` :    [ float ]   radians, positive = left
    `speed` :       [ float ]   rad/s
    `parked` :      [ bool ]    True when stopped in the middle of the bay
    '''
    global _last_steering, _lost_frames, _last_bearing

    if pose is None:
        return _marker_lost()
    _lost_frames = 0
    _last_bearing = pose['bearing']

    if pose['forward'] <= BAY_MIDDLE_FORWARD + STOP_LEAD:
        return 0.0, 0.0, True

    steering = _path_steering(pose)
    _last_steering = steering
    return steering, _speed_for(pose['forward']), False


def _path_steering(pose):
    '''Steer to hold the heading that brings the vehicle onto the centre line in time.'''
    lateral = pose['lateral']       # m, + = left of the centre line (front axle)
    heading = pose['heading']       # rad, + = turned left

    # Distance left to run before the vehicle must be on the line and straight.
    plan_range = max(pose['forward'] - PLAN_END_FORWARD, MIN_PLAN_RANGE)

    # Heading wanted right now: a smooth crossing is about PATH_GAIN times steeper than the
    # straight line from here to the line, so aim that steep - but never past the cap.
    # It fades to zero as the offset does, which is what straightens the vehicle out.
    wanted = -np.arctan(PATH_GAIN * lateral / plan_range)
    wanted = float(np.clip(wanted, -HEADING_CAP, HEADING_CAP))

    # The marker is about to leave the camera view: stop turning further that way.
    if abs(pose['bearing']) > BEARING_LIMIT and wanted * pose['bearing'] < 0.0:
        wanted = max(wanted, heading) if wanted < 0.0 else min(wanted, heading)

    steering = STEER_GAIN * (wanted - heading)
    return float(np.clip(steering, -MAX_STEERING, MAX_STEERING))


def _speed_for(forward):
    '''Full speed far out, easing down to a slow creep for the last stretch.'''
    return float(np.interp(forward, [CREEP_FROM, CRUISE_UNTIL], [CREEP_SPEED, CRUISE_SPEED]))


def _marker_lost():
    '''A missed frame is normal: hold the last steering, creeping. If it persists, turn back
    towards where the marker was last seen (it slides out of view when we are turned too
    far), and if that fails, stop.'''
    global _lost_frames
    _lost_frames += 1
    if _lost_frames <= HOLD_FRAMES:
        return _last_steering, CREEP_SPEED, False
    if _lost_frames <= GIVE_UP_FRAMES:
        return float(np.sign(_last_bearing) * SEARCH_STEERING), CREEP_SPEED, False
    return 0.0, 0.0, False

##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera, finding the marker, working
# out the vehicle's pose relative to the bay, and writing to the joints. None
# of it involves a decision you are marked on.
#
#####################################################################

WHEELBASE      = 0.120     # L, metres
TRACK_WIDTH    = 0.110     # W, metres
WHEEL_OFFSET   = 0.0275    # O, metres
STEER_LIMIT    = np.deg2rad(45.0)

# ---- camera and marker, as in Task_2C.ttt ----
IMG_W, IMG_H  = 640, 480
FOCAL_PX      = (IMG_W / 2) / np.tan(np.deg2rad(95.0) / 2)   # 95 deg horizontal FOV
CAMERA_MATRIX = np.array([[FOCAL_PX, 0, IMG_W / 2],
                          [0, FOCAL_PX, IMG_H / 2],
                          [0, 0, 1]])
CAM_PITCH     = np.deg2rad(30.0)     # camera tilted 30 deg down
CAM_ABOVE_MK  = 0.176 - 0.110        # camera height minus marker-centre height, m
CAM_AHEAD     = 0.047                # camera is this far ahead of the vehicle centre
# The black pattern, not the board: the 0.12 m board has a white border and
# the pattern is 80% of it.
MARKER_SIDE   = 0.096
_H = MARKER_SIDE / 2
MARKER_CORNERS_3D = np.array([[-_H, _H, 0], [_H, _H, 0],
                              [_H, -_H, 0], [-_H, -_H, 0]], np.float32)

# ---- pose filter ----
YAW_BLEND       = 0.12               # how far each marker reading pulls the heading
YAW_GATE        = np.deg2rad(15.0)   # readings further than this from the prediction are rejected
AMBIGUITY_RATIO = 1.5                # PnP's two answers must differ this much in fit to trust one


def ackermann_wheel_angles(delta):
    '''Split one steering angle into the two front-wheel angles.

    This is Task 1A's geometry, provided here so that Task 2C is only about
    the manoeuvre. The kingpins sit WHEEL_OFFSET inboard of each wheel
    centre, so the triangle is built on the kingpin-to-kingpin width.
    '''
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    l = WHEELBASE
    w = TRACK_WIDTH - 2 * WHEEL_OFFSET
    s, c = np.sin(delta), np.cos(delta)
    left = np.arctan2(2 * l * s, 2 * l * c - w * s)
    right = np.arctan2(2 * l * s, 2 * l * c + w * s)
    return float(left), float(right)


def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.'''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2C.ttt open?\n'
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
                f'  Open Task_2C.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    Three corrections turn the sensor's raw buffer into an image, and none of
    the three announces itself if you leave it out:

      * handleVisionSensor() - the sensor renders only when asked.
      * cv2.flip(frame, 0)   - the buffer's first row is the BOTTOM row.
      * RGB2BGR              - the buffer is RGB; OpenCV assumes BGR.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def find_marker(detector, frame, target_id):
    '''The target marker's four corners in the image (top-left, top-right,
    bottom-right, bottom-left), or None when it is not in this frame.'''
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    for quad, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) == int(target_id):
            return quad.reshape(4, 2)
    return None


class PoseEstimator:
    '''Where the vehicle is relative to the target bay, from the marker.

    Position comes straight from the marker each frame. Heading is harder: a
    small marker far away gives a noisy reading, and a square seen nearly
    face-on has two mirror-image poses that fit its corners almost equally
    well. So heading is predicted from the steering actually commanded, and
    each clear marker reading pulls that prediction a little toward itself.
    The vehicle is spawned parallel to the bays, which is where it starts.
    '''

    def __init__(self):
        self.heading = 0.0      # rad, + = turned RIGHT (internal convention)
        self.rho = None         # smoothed horizontal range to the marker, m

    @staticmethod
    def _bearing(corners):
        '''Horizontal bearing to the marker centre, rad, + = to the RIGHT.'''
        cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
        xc = (cx - IMG_W / 2) / FOCAL_PX
        yc = (IMG_H / 2 - cy) / FOCAL_PX
        return float(np.arctan2(xc, np.cos(CAM_PITCH) + yc * np.sin(CAM_PITCH)))

    @staticmethod
    def _pnp(corners):
        '''Heading (rad, + = RIGHT, or None if ambiguous) and horizontal range (m).'''
        _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            MARKER_CORNERS_3D, corners.astype(np.float32), CAMERA_MATRIX,
            np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = []
        for rv, tv, e in zip(rvecs, tvecs, np.array(errs).flatten()):
            R, _ = cv2.Rodrigues(rv)
            fwd = R.T @ np.array([0.0, 0.0, 1.0])   # camera forward, in the marker frame
            sols.append((float(e), float(np.arctan2(fwd[0], -fwd[2])), tv.flatten()))
        sols.sort(key=lambda s: s[0])
        t = sols[0][2]
        rho = float(np.sqrt(max(t @ t - CAM_ABOVE_MK ** 2, 1e-6)))
        clear = len(sols) < 2 or sols[1][0] > AMBIGUITY_RATIO * max(sols[0][0], 1e-6)
        return (sols[0][1] if clear else None), rho

    def update(self, corners, last_steering):
        '''The pose handed to parking_manoeuvre(), or None if no marker.'''
        if corners is None:
            return None
        bearing = self._bearing(corners)
        yaw_seen, rho_now = self._pnp(corners)
        if self.rho is None:
            self.rho = rho_now
        rho_prev, self.rho = self.rho, 0.6 * self.rho + 0.4 * rho_now
        # Distance covered since the last frame: range shrinks at cos(bearing)
        # per metre travelled.
        ds = min(max((rho_prev - self.rho) / max(np.cos(bearing), 0.3), 0.0), 0.01)
        # Steering LEFT (+) turns the heading LEFT, NEGATIVE in this convention.
        self.heading -= np.tan(last_steering) / WHEELBASE * ds
        if yaw_seen is not None and abs(yaw_seen - self.heading) < YAW_GATE:
            self.heading += YAW_BLEND * (yaw_seen - self.heading)

        los = bearing + self.heading                # line of sight, from the bay axis
        lateral_cam = -self.rho * np.sin(los)       # + = RIGHT of the centre line
        forward_cam = self.rho * np.cos(los)
        lateral_front = lateral_cam + (WHEELBASE / 2 - CAM_AHEAD) * np.sin(self.heading)
        forward = forward_cam + CAM_AHEAD * np.cos(self.heading)
        # Handed over in the Task 1 convention: POSITIVE = LEFT throughout.
        return {
            'lateral': float(-lateral_front),
            'heading': float(-self.heading),
            'forward': float(forward),
            'bearing': float(-bearing),
        }


def drive(sim, handles, steering, speed):
    '''Split the steering and write it, with the drive speed.'''
    left, right = ackermann_wheel_angles(steering)
    sim.setJointTargetPosition(handles['steer_left'], left)
    sim.setJointTargetPosition(handles['steer_right'], right)
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def parse_args():
    '''The bay to park in, from the command line - or TARGET_ID if not given.'''
    parser = argparse.ArgumentParser(
        description='Task 2C - park the vehicle in one of the two bays.')
    parser.add_argument('--bay', type=int, choices=(0, 1), default=None,
                        help='which bay to park in (default: TARGET_ID in this file)')
    return parser.parse_args()


def main():
    global TARGET_ID
    # Read the flag before touching the simulator, so --help and a bad
    # value are answered straight away.
    args = parse_args()
    if args.bay is not None:
        TARGET_ID = args.bay

    sim, handles = connect_and_locate()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters())
    estimator = PoseEstimator()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, spawns the vehicle and starts the simulation,
    # so wait for that rather than starting your own.
    deadline = time.monotonic() + 120.0
    if sim.getSimulationState() != sim.simulation_advancing_running:
        print('Waiting for the simulation to start - press Play in CoppeliaSim.')
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - parking in bay %d.' % TARGET_ID)

    frames, seen, parked, steering = 0, 0, False, 0.0
    try:
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            corners = find_marker(detector, frame, TARGET_ID)
            if corners is not None:
                seen += 1
            pose = estimator.update(corners, steering)

            steering, speed, parked = parking_manoeuvre(pose)
            if parked:
                drive(sim, handles, 0.0, 0.0)
                print('Parked after %d frames.' % frames)
                break
            steering = float(np.clip(steering, -STEER_LIMIT, STEER_LIMIT))
            drive(sim, handles, steering, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        try:
            drive(sim, handles, 0.0, 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed, marker {TARGET_ID} seen in {seen}.')
    if not parked:
        print('The run ended without parking_manoeuvre() returning parked=True.')
    return 0


if __name__ == '__main__':
    sys.exit(main())