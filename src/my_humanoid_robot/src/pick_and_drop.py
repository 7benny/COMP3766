#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA

def interpolate_poses(start_pos, end_pos, steps):
    """Interpolate between two positions over a given number of steps."""
    rospy.loginfo("Interpolating poses: start=%s, end=%s, steps=%d", start_pos, end_pos, steps)
    return [start_pos + (end_pos - start_pos) * i / (steps - 1) for i in range(steps)]

def publish_object_marker(position, marker_pub, action):
    """Publish a marker to represent the object being picked up or dropped."""
    marker = Marker()
    marker.header = Header(frame_id="base_link", stamp=rospy.Time.now())
    marker.ns = "object"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = action 
    marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1 
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) 
    marker.lifetime = rospy.Duration(0.1)  
    rospy.loginfo("Publishing object marker: action=%d, position=%s", action, position)
    marker_pub.publish(marker)

def publish_pick_and_drop():
    rospy.loginfo("Starting pick_and_drop_node initialization")
    rospy.init_node("pick_and_drop_node", anonymous=True)
    rospy.loginfo("Node initialized")

    rospy.loginfo("Setting up publishers")
    left_pub = rospy.Publisher("/left_goal_pose", Pose, queue_size=10)
    right_pub = rospy.Publisher("/right_goal_pose", Pose, queue_size=10)
    object_marker_pub = rospy.Publisher("/object_marker", Marker, queue_size=10)
    rospy.loginfo("Publishers set up")

    rate = rospy.Rate(10)
    rospy.loginfo("Rate set to 10 Hz")

    neutral_left = np.array([0.3, 0.2, 0.3])
    neutral_right = np.array([-0.3, 0.2, 0.3])
    pick_positions = [np.array([0.0, 0.35, 0.4]), np.array([-0.2, 0.35, 0.4])]
    drop_positions = [np.array([0.3, 0.4, 0.6]), np.array([-0.2, 0.35, 0.5])]
    rospy.loginfo("Positions defined: neutral_left=%s, neutral_right=%s", neutral_left, neutral_right)
    rospy.loginfo("Pick positions: %s", pick_positions)
    rospy.loginfo("Drop positions: %s", drop_positions)

    steps = 20 
    hold_steps = 10 
    rospy.loginfo("Steps=%d, hold_steps=%d", steps, hold_steps)

    current_left_pos = neutral_left
    current_right_pos = neutral_right
    rospy.loginfo("Initial positions: current_left_pos=%s, current_right_pos=%s", current_left_pos, current_right_pos)

    rospy.loginfo("Starting multiple pick-and-drop tasks")

    for idx, (pick_pos, drop_pos) in enumerate(zip(pick_positions, drop_positions)):
        rospy.loginfo("Starting pick-and-drop sequence %d: pick_pos=%s, drop_pos=%s", idx + 1, pick_pos, drop_pos)

        left_to_pick = interpolate_poses(current_left_pos, pick_pos, steps)
        right_to_pick = interpolate_poses(current_right_pos, pick_pos, steps)
        pick_to_drop = interpolate_poses(pick_pos, drop_pos, steps)
        left_drop_to_neutral = interpolate_poses(drop_pos, neutral_left, steps)
        right_drop_to_neutral = interpolate_poses(drop_pos, neutral_right, steps)
        rospy.loginfo("Interpolation completed for sequence %d", idx + 1)

        left_sequence = (left_to_pick + [pick_pos] * hold_steps +
                         pick_to_drop + [drop_pos] * hold_steps +
                         left_drop_to_neutral)
        right_sequence = (right_to_pick + [pick_pos] * hold_steps +
                          pick_to_drop + [drop_pos] * hold_steps +
                          right_drop_to_neutral)
        object_sequence = ([None] * steps + [pick_pos] * hold_steps +
                           pick_to_drop + [drop_pos] * hold_steps +
                           [None] * steps)
        rospy.loginfo("Sequences combined for sequence %d: len(left_sequence)=%d", idx + 1, len(left_sequence))

        # Publish the sequence
        for i, (left_pos, right_pos, obj_pos) in enumerate(zip(left_sequence, right_sequence, object_sequence)):
            left_pose = Pose()
            left_pose.position.x = left_pos[0]
            left_pose.position.y = left_pos[1]
            left_pose.position.z = left_pos[2]
            left_pose.orientation.w = 1.0

            right_pose = Pose()
            right_pose.position.x = right_pos[0]
            right_pose.position.y = right_pos[1]
            right_pose.position.z = right_pos[2]
            right_pose.orientation.w = 1.0

            rospy.loginfo("Publishing poses at step %d: left_pos=%s, right_pos=%s", i, left_pos, right_pos)
            left_pub.publish(left_pose)
            right_pub.publish(right_pose)

            if obj_pos is not None:
                rospy.loginfo("Publishing object marker ADD at position %s", obj_pos)
                publish_object_marker(obj_pos, object_marker_pub, Marker.ADD)
            else:
                rospy.loginfo("Publishing object marker DELETE")
                publish_object_marker([0, 0, 0], object_marker_pub, Marker.DELETE)

            rate.sleep()
            rospy.sleep(0.5) 

        current_left_pos = neutral_left
        current_right_pos = neutral_right
        rospy.loginfo("Completed sequence %d, resetting to neutral positions", idx + 1)

    publish_object_marker([0, 0, 0], object_marker_pub, Marker.DELETE)
    rospy.loginfo("Multiple pick-and-drop tasks completed")

if __name__ == "__main__":
    try:
        rospy.loginfo("Starting main execution")
        publish_pick_and_drop()
        rospy.loginfo("Main execution completed")
    except rospy.ROSInterruptException:
        rospy.loginfo("ROSInterruptException caught")
    except Exception as e:
        rospy.logerr("Unexpected error in pick_and_drop.py: %s", str(e))