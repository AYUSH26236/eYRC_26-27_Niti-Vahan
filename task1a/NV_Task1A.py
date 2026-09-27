'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1A of Niti Vahan (NV) Theme.
*
*  Filename:         ackermann_steering.py
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
# Author List:      <Ayush Tiwari, Anushka Telore , Atharva Jadhav , Utkarsh Singh >
# Filename:         ackermann_steering.py
# Functions:        ackermann_wheel_angles
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import math
import numpy as np
##############################################################


#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################


'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1A of Niti Vahan (NV) Theme.
*
*  Filename:         ackermann_steering.py
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
# Author List:      <Ayush Tiwari, Anushka Telore , Atharva Jadhav , Utkarsh Singh >
# Filename:         ackermann_steering.py
# Functions:        ackermann_wheel_angles
# Global variables: < None >


####################### IMPORT MODULES #######################
import math
import numpy as np
##############################################################


#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################


def ackermann_wheel_angles(delta):
    '''
    Purpose:
    ---
    Convert a single virtual steering angle into the two real front-wheel
    angles, per the Ackermann geometry.

    Input Arguments:
    ---
    `delta` :   [ float ]
        Steering angle of the virtual centred front wheel, in radians.

    Returns:
    ---
    `left_angle`  : [ float ]
        The left front wheel steering angle in radians.

    `right_angle` : [ float ]
        The right front wheel steering angle in radians.
    '''

    # When there is no steering, both wheels remain straight

    if abs(delta) < 1e-9:

        return 0.0, 0.0

    # Half of the track width, including the wheel offset

    half_track = TRACK_WIDTH / 2.0 - WHEEL_OFFSET

    # Radius of the path followed by the virtual centre wheel

    radius = WHEELBASE / math.tan(abs(delta))

    # Angle of the inside wheel

    inside_angle = math.atan(

        WHEELBASE / (radius - half_track)

    )

    # Angle of the outside wheel

    outside_angle = math.atan(

        WHEELBASE / (radius + half_track)

    )

    # Positive delta means turning left

    if delta > 0:

        left_angle = inside_angle

        right_angle = outside_angle

    # Negative delta means turning right

    else:

        left_angle = -outside_angle

        right_angle = -inside_angle

    return left_angle, right_angle



##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

if __name__ == "__main__":

    test_angles = np.arange(-0.35, 0.35, 0.05)

    for d in test_angles:
        left, right = ackermann_wheel_angles(d)
        print(f"delta={d}  ->  left={left}, right={right}")
