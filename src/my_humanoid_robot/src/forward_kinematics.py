#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState

def expm_so3(w, theta):
    """Compute the exponential map for SO(3) (rotation matrix) given an axis w and angle theta."""
    w_skew = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]])
    return np.eye(3) + np.sin(theta) * w_skew + (1 - np.cos(theta)) * w_skew @ w_skew

def homogeneous_transform(R, p):
    """Construct a 4x4 homogeneous transformation matrix from rotation R and translation p."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def forward_kinematics(joint_angles):
    """Compute the forward kinematics of the left arm using homogeneous transformations."""
    # Joint angles
    theta1 = joint_angles[0]  
    theta2 = joint_angles[1]  
    theta3 = joint_angles[2]  

    shoulder_offset = np.array([0.15, 0, 0.25]) 
    upper_arm_length = 0.3 
    lower_arm_length = 0.25  

    T_torso_to_shoulder = homogeneous_transform(np.eye(3), shoulder_offset)

    # Joint 1: left_shoulder_yaw (rotation around Z-axis)
    w1 = np.array([0, 0, 1])  # Axis of rotation
    R1 = expm_so3(w1, theta1)
    T_shoulder_yaw = homogeneous_transform(R1, np.array([0, 0, 0]))

    # Joint 2: left_shoulder_pitch (rotation around Y-axis)
    w2 = np.array([0, 1, 0])  # Axis of rotation
    R2 = expm_so3(w2, theta2)
    T_shoulder_pitch = homogeneous_transform(R2, np.array([0, 0, 0]))

    T_upper_arm = homogeneous_transform(np.eye(3), np.array([0, 0, -upper_arm_length]))

    # Joint 3: left_elbow (rotation around Y-axis)
    w3 = np.array([0, 1, 0])  # Axis of rotation
    R3 = expm_so3(w3, theta3)
    T_elbow = homogeneous_transform(R3, np.array([0, 0, 0]))

    T_lower_arm = homogeneous_transform(np.eye(3), np.array([0, 0, -lower_arm_length]))

    T_total = T_torso_to_shoulder @ T_shoulder_yaw @ T_shoulder_pitch @ T_upper_arm @ T_elbow @ T_lower_arm

    end_pos = T_total[:3, 3]
    return end_pos

def joint_state_callback(msg):
    """Callback to process joint states and compute FK for the left arm."""
    try:
        yaw_idx = msg.name.index("left_shoulder_yaw")
        pitch_idx = msg.name.index("left_shoulder_pitch")
        elbow_idx = msg.name.index("left_elbow")
        joint_angles = [msg.position[yaw_idx], msg.position[pitch_idx], msg.position[elbow_idx]]
        end_pos = forward_kinematics(joint_angles)
        rospy.loginfo("Left arm end position: x=%.2f, y=%.2f, z=%.2f", end_pos[0], end_pos[1], end_pos[2])
    except ValueError:
        rospy.logwarn("Joint names not found in JointState message")

def main():
    rospy.init_node("forward_kinematics_node", anonymous=True)
    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
