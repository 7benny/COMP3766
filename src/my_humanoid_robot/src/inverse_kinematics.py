#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
import tf
from tf.transformations import translation_from_matrix, quaternion_matrix

def compute_ik(position, orientation, listener, arm):
    """
    Analytical IK for the specified arm (3 DOF: shoulder yaw, shoulder pitch, elbow).
    Takes a 3x1 position vector, a 3x3 rotation matrix (orientation ignored),
    a TF listener to get the shoulder position, and the arm ("left" or "right").
    Returns a list of joint positions.
    """
    print(f"\nReceived Position for {arm} arm:")
    print(position)

    print(f"\nReceived Orientation for {arm} arm (3x3 Rotation Matrix):")
    print(orientation)

    shoulder_frame = f"{arm}_shoulder_yaw"
    try:
        (trans, rot) = listener.lookupTransform('base_link', shoulder_frame, rospy.Time(0))
        shoulder_offset = np.array(trans)  # [x, y, z] position of shoulder
        rospy.loginfo(f"{arm.capitalize()} shoulder position from TF: %s", shoulder_offset)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn("TF lookup failed: %s", e)
        shoulder_offset = np.array([0.15 if arm == "left" else -0.15, 0, 0.75])
        rospy.logwarn(f"Using hardcoded {arm} shoulder position: %s", shoulder_offset)

    # Robot parameters (from URDF)
    a2 = 0.3  # Upper arm length
    a3 = 0.25  # Lower arm length

    # Extract target position
    px, py, pz = position

    # Step 1: Compute theta1 (shoulder_yaw)
    r = np.sqrt((px - shoulder_offset[0])**2 + (py - shoulder_offset[1])**2)
    theta1 = np.arctan2(py - shoulder_offset[1], px - shoulder_offset[0])  

    # Step 2: Compute the distance from shoulder to target in the YZ plane
    s = pz - shoulder_offset[2]  
    d = np.sqrt(r**2 + s**2) 

    # Step 3: Compute theta3 (elbow) using the law of cosines
    D = (d**2 - a2**2 - a3**2) / (2 * a2 * a3)
    D = np.clip(D, -1.0, 1.0)  
    theta3 = -np.arccos(D)

    # Step 4: Compute theta2 (shoulder_pitch)
    phi = np.arctan2(s, r)
    psi = np.arctan2(a3 * np.sin(-theta3), a2 + a3 * np.cos(theta3))
    theta2 = phi + psi

    joint_limits = {
        'shoulder_yaw': [-1.57, 1.57],
        'shoulder_pitch': [-1.57, 1.57],
        'elbow': [-1.57, 0]
    }
    theta1 = np.clip(theta1, *joint_limits['shoulder_yaw'])
    theta2 = np.clip(theta2, *joint_limits['shoulder_pitch'])
    theta3 = np.clip(theta3, *joint_limits['elbow'])

    # Check reachability
    max_reach = a2 + a3
    if d > max_reach:
        rospy.logwarn(f"{arm.capitalize()} arm: Target is out of reach! Distance: %.2f, Max reach: %.2f", d, max_reach)

    return np.array([theta1, theta2, theta3])

def publish_target_marker(position, marker_pub, arm):
    """Publish a marker to visualize the target position in RViz."""
    marker = Marker()
    marker.header = Header(frame_id="base_link", stamp=rospy.Time.now())
    marker.ns = f"{arm}_target"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color = ColorRGBA(r=1.0 if arm == "left" else 0.0, g=0.0, b=0.0 if arm == "left" else 1.0, a=1.0)  # Red for left, blue for right
    marker.lifetime = rospy.Duration(0) 
    marker_pub.publish(marker)

class IKSolver:
    def __init__(self):
        self.listener = tf.TransformListener()

        self.joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.marker_pub = rospy.Publisher("/target_marker", Marker, queue_size=10)

        self.left_sub = rospy.Subscriber("/left_goal_pose", Pose, self.left_pose_callback)
        self.right_sub = rospy.Subscriber("/right_goal_pose", Pose, self.right_pose_callback)

        self.joint_positions = {
            'left_shoulder_yaw': 0.0,
            'left_shoulder_pitch': 0.0,
            'left_elbow': 0.0,
            'right_shoulder_yaw': 0.0,
            'right_shoulder_pitch': 0.0,
            'right_elbow': 0.0
        }

    def left_pose_callback(self, msg):
        self.handle_pose_callback(msg, "left")

    def right_pose_callback(self, msg):
        self.handle_pose_callback(msg, "right")

    def handle_pose_callback(self, msg, arm):
        """
        Callback function to handle incoming end-effector pose messages for the specified arm.
        """
        position = np.array([msg.position.x, msg.position.y, msg.position.z])

        q = msg.orientation
        orientation = np.array([
            [1 - 2 * (q.y**2 + q.z**2), 2 * (q.x*q.y - q.z*q.w), 2 * (q.x*q.z + q.y*q.w)],
            [2 * (q.x*q.y + q.z*q.w), 1 - 2 * (q.x**2 + q.z**2), 2 * (q.y*q.z - q.x*q.w)],
            [2 * (q.x*q.z - q.y*q.w), 2 * (q.y*q.z + q.x*q.w), 1 - 2 * (q.x**2 + q.y**2)]
        ])

        publish_target_marker(position, self.marker_pub, arm)

        joint_positions = compute_ik(position, orientation, self.listener, arm)

        rospy.loginfo(f"Computed joint angles for {arm} arm: {joint_positions}")
        self.joint_positions[f"{arm}_shoulder_yaw"] = joint_positions[0]
        self.joint_positions[f"{arm}_shoulder_pitch"] = joint_positions[1]
        self.joint_positions[f"{arm}_elbow"] = joint_positions[2]

        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [
            'left_shoulder_yaw', 'left_shoulder_pitch', 'left_elbow',
            'right_shoulder_yaw', 'right_shoulder_pitch', 'right_elbow'
        ]
        joint_msg.position = [
            self.joint_positions['left_shoulder_yaw'],
            self.joint_positions['left_shoulder_pitch'],
            self.joint_positions['left_elbow'],
            self.joint_positions['right_shoulder_yaw'],
            self.joint_positions['right_shoulder_pitch'],
            self.joint_positions['right_elbow']
        ]
        rospy.loginfo(f"Publishing joint states: {joint_msg.position}")
        self.joint_pub.publish(joint_msg)

if __name__ == "__main__":
    rospy.init_node("ik_solver_node", anonymous=True)
    ik_solver = IKSolver()
    print("\nWaiting for /left_goal_pose and /right_goal_pose")
    rospy.spin()