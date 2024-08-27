#!/usr/bin/env python3
# ROS dependencies
from os import times_result
import rospy
import math
import scipy.io as scio
import numpy as np

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

class distCalculation:
    def __init__(self):
        
        self.odom_subscriber_ = rospy.Subscriber("/zed2i/zed_node/odom", Odometry, self.distance_callback)
        self.path_publisher = rospy.Publisher("/e2e_nav/path", Path, queue_size=10)

        self.path_  = Path()
        self.path_dist_ = 0
        self.init_ = 0
        self.goal_ = Odometry()
        self.goal_subscriber_ = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

        self.record_ = 0

        self.time_list, self.x_list, self.y_list = [], [], []

    def distance_callback(self, odom_msg):
        if self.record_ == 0:
            return
        cur_pose = PoseStamped()
        cur_pose.header = odom_msg.header
        cur_pose.pose = odom_msg.pose.pose
        time = rospy.get_rostime().to_sec()
        # print("current time: ", time )

        if self.init_ == 0:
            self.path_.poses.append(cur_pose)
            self.init_ = 1
            return

        last_pose = self.path_.poses[-1]
        
        diff = math.sqrt((cur_pose.pose.position.x - last_pose.pose.position.x)**2 + (cur_pose.pose.position.y - last_pose.pose.position.y)**2 )
        self.path_dist_ +=diff
        self.time_list.append(time)
        self.x_list.append(cur_pose.pose.position.x)
        self.y_list.append(cur_pose.pose.position.y)

        self.path_.header = odom_msg.header
        self.path_.poses.append(cur_pose)
        self.path_publisher.publish(self.path_)

        if (cur_pose.pose.position.x - 5.935)**2 + (cur_pose.pose.position.y - (-4.733))**2 < 0.5:
            print("Goal Reached!")
            print ("Using time: ", self.time_list[-1] - self.time_list[0])
            print ("Path travled: ", self.path_dist_)
            scio.savemat('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/data/data_record{}.mat'.format(rospy.get_rostime().to_sec()), mdict={'t':self.time_list, 'x':self.x_list, 'y':self.y_list})
            rospy.signal_shutdown("Data recording completed!")
        # print ("The distance of path traveled is : " + str(self.path_dist_))

    def goal_callback(self, goal_msg):
        print("Goal Recieved")
        self.record_ = 1
        # print(goal_msg.pose.position.x, goal_msg.pose.position.y)
        # print ("Using time: ", self.path_.poses[-1].header.stamp - self.path_.poses[0].header.stamp)
        # self.init_ = 0
        # self.path_.poses = []
        # self.path_dist_ = 0
        
        # # self.time_list = np.array(self.time_list)
        # # self.x_list = np.array(self.x_list)
        # # self.y_list = np.array(self.y_list)
        # scio.savemat('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/data_record{}.mat'.format(rospy.get_rostime().to_sec()), mdict={'t':self.time_list, 'x':self.x_list, 'y':self.y_list})
        # # del self.time_list
        # # del self.x_list
        # # del self.y_list
        # self.time_list, self.x_list, self.y_list = [], [], []

if __name__ == '__main__':
    print("Path traveled distance calculator!")
    rospy.init_node('distance_calculator')
    distCalculation()
    rospy.spin()
