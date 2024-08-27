#!/usr/bin/env python3
# ROS dependencies
import queue
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from rospy.numpy_msg import numpy_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy

import message_filters

# RL dependencies
import numpy as np
import torch
from SAC_based_DRL_denoise.sac import DRL


# from darknet_ros_msgs.msg import BoundingBox
# from darknet_ros_msgs.msg import BoundingBoxes

#Deque
from collections import deque
import math
from squaternion import Quaternion

#Keyboard
import getch

class e2eNavigation:
    def __init__(self):
        # DRL model related
        self.agent_ = DRL(action_dim=2, pstate_dim=3, BUFFER_SIZE=1e0, LOAD_ENCODER=1, FREEZE_ENCODER=1)
        # self.agent_.policy.load_state_dict(torch.load('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/actor1007.pkl', map_location=torch.device('cpu')))
        self.agent_.policy.load_state_dict(torch.load('/home/yanxin/phd_workspace/e2e_nav/scripts/models/actor1007.pkl', map_location=torch.device('cpu')))
        self.agent_.critic.load_state_dict(torch.load('/home/yanxin/phd_workspace/e2e_nav/scripts/models/critic1007.pkl', map_location=torch.device('cpu')))
        
        # define ROS subscriber and pulisher
        self.semantic_subscriber_ = message_filters.Subscriber("/segnet/color_mask", numpy_msg(Image))
        # self.bbx_subscriber_ = message_filters.Subscriber("darknet_ros/bounding_boxes", BoundingBoxes)
        self.odom_subscriber_ = message_filters.Subscriber("/odom_aloam_high_freq", Odometry)
        
        # sync odom from slam and semantic segmentation images
        ts = message_filters.ApproximateTimeSynchronizer([self.semantic_subscriber_, self.odom_subscriber_], 100, 0.1, allow_headerless=False)
        ts.registerCallback(self.semantic_callback)

        self.takeover_trigger_subscriber = rospy.Subscriber("/takeover", Int64, self.takeover_listener_callback)
        self.twistPublisher_ = rospy.Publisher("/hunter/cmd_vel", Twist, queue_size=30)
        self.img_publisher_ = rospy.Publisher("/automan/seg_img", Image, queue_size=10 )
        self.path_publisher = rospy.Publisher("/e2e_nav/path", Path, queue_size=10)

        self.e2e_path_  = Path()

        self.goal_ = Odometry()
        self.goal_subscriber_ = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        
        self.end_x_ = 5.0 #5.935
        self.end_y_ = 0.0 #-4.743
        self.bridge_ = CvBridge()

        self.init_ = False # the image deque is not fullfilled at beginning
        self.takeover_ = 0 # flag to take over agv

        self.takeover_deque_ = deque(maxlen = 2)    # once button is touched, the agv would be taken over
        for i in range(2):
            self.takeover_deque_.append(0)
        
        self.yaw_ego_ = 0 
        self.image_deque_ = deque(maxlen = 3)
        self.p_state_ = np.array([0.0, 0.0, 0.0]) # p_state: 1. yaw of ego motion 
                                            # 2. distance to goal point 
                                            # 3. the angle related to goal point

        self.img_state_ = np.repeat(np.expand_dims(np.zeros(shape=(90, 180)),2), 3, axis=2) # composition of 3 images
        self.alpha_v_ = 0.6
        print("System Initialized! When drivable area imge showing , the inference starts!")

    def semantic_callback(self, semantic_msg, odom_msg):
        try:
            img = np.float32(ros_numpy.numpify(semantic_msg))
            tmp = np.ones((img.shape[0], img.shape[1])).astype(np.float32)
            # tmp[np.where((img[:,:,0]==0.0) & (img[:,:,1]==128.0) & (img[:,:,2]==0.0))]=0.05
            # tmp[np.where((img[:,:,0]==192.0) & (img[:,:,1]==192.0) & (img[:,:,2]==128.0))]=0.55
            # tmp[np.where((img[:,:,0]==128.0) & (img[:,:,1]==0.0) & (img[:,:,2]==128.0))]=0.55

            tmp[np.where((img[:,:,0]==0.0) & (img[:,:,1]==184.0) & (img[:,:,2]==255.0))]=0.05
            tmp[np.where((img[:,:,0]==133.0) & (img[:,:,1]==0.0) & (img[:,:,2]==255.0))]=0.05
            # tmp[np.where((img[:,:,0]==140.0) & (img[:,:,1]==140.0) & (img[:,:,2]==140.0))]=0.05
            tmp[np.where((img[:,:,0]==112.0) & (img[:,:,1]==9.0) & (img[:,:,2]==255.0))]=0.05
            tmp[np.where((img[:,:,0]==4.0) & (img[:,:,1]==250.0) & (img[:,:,2]==7.0))]=0.05
            tmp[np.where((img[:,:,0]==80.0) & (img[:,:,1]==50.0) & (img[:,:,2]==50.0))]=0.05

            tmp[np.where((img[:,:,0]==150.0) & (img[:,:,1]==5.0) & (img[:,:,2]==61.0))]=0.55
            tmp[np.where((img[:,:,0]==10.0) & (img[:,:,1]==0.0) & (img[:,:,2]==255.0))]=0.55
            
            # for bbx in bbx_msg.bounding_boxes:
            #     if bbx.Class =="person" and bbx.probability > 0.8:
            #         tmp[bbx.ymin : bbx.ymax, bbx.xmin : bbx.xmax] = 0.55


            img1 = cv2.resize(tmp, (180, 90))
            semantic_img =img1
            # img1 = cv2.medianBlur(img1, 5)
            # kernel = np.ones((3, 3), np.uint8)
            # semantic_img = cv2.dilate(img1, kernel) 
            # semantic_img = cv2.GaussianBlur(semantic_img, (5,5), 0)


            self.img_publisher_.publish(self.bridge_.cv2_to_imgmsg(semantic_img))
            
            # cv2.namedWindow("drivable area", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('drivable area', 640, 480) 
            # cv2.imshow('drivable area', semantic_img)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

        self.image_deque_.append(np.array(semantic_img))
        if len(self.image_deque_)==3:
            self.init_ = True

        # goal is set to agv it self at beginning
        if (not self.init_):
            self.goal_.pose.pose.position.x = odom_msg.pose.pose.position.x
            self.goal_.pose.pose.position.y = odom_msg.pose.pose.position.y
        

        velMsg = Twist()

        cur_pose = PoseStamped()
        cur_pose.header = odom_msg.header
        cur_pose.pose = odom_msg.pose.pose

        self.e2e_path_.header = odom_msg.header
        # self.e2e_path_.header.frame_id = ''
        self.e2e_path_.poses.append(cur_pose)

        self.path_publisher.publish(self.e2e_path_)

        # goal_state calculation
        odomX = odom_msg.pose.pose.position.x
        odomY = odom_msg.pose.pose.position.y
        goalX = self.goal_.pose.pose.position.x
        goalY = self.goal_.pose.pose.position.y

        quaternion = Quaternion(
            odom_msg.pose.pose.orientation.w,
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        heading_angle = round(euler[2], 4)

        self.yaw_ego_ = heading_angle

        # Calculate the angle distance between the robots heading and heading toward the goal
        skewX = goalX - odomX
        skewY = goalY - odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        if(mag1 < 1e-1 ):
            print("Please give a 2D Nav Goal! " + "The dist_l2 is: " + str(mag1))
            return
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - heading_angle)

        # jingda version
        if beta2 > np.pi:
            beta2 = (beta2 - 2*np.pi)/np.pi
        elif beta2 < -np.pi:
            beta2 = (beta2 + 2*np.pi)/np.pi
        else:
            beta2 = beta2/np.pi

        beta2 = -beta2
        # if beta2 > np.pi:
        #     beta2 = np.pi - beta2
        #     beta2 = -np.pi - beta2
        # if beta2 < -np.pi:
        #     beta2 = -np.pi - beta2
        #     beta2 = np.pi - beta2

        # Calculate distance to the goal from the robot
        dist_l2 = math.sqrt(math.pow(odomX - goalX, 2) + math.pow(odomY - goalY, 2))   
        print("The dist_l2 is: " + str(mag1))     
        if dist_l2 < 0.2:
            print("Goal Reached!")
            print("The dist_l2 is: " + str(mag1))
            print("Please give a new goal point!")
            velMsg.linear.x = 0
            velMsg.angular.z = 0
            self.twistPublisher_.publish(velMsg)
            return

        # jingda version: normalization
        rho = dist_l2/13

        if (self.init_ & (not self.takeover_)):
        
            action = self.inference()
            # action[0] = 0 if action[0]<0 else action[0]
            action[0] = np.clip(action[0]*0.75+0.25,0,1)
			#print(action)
            # self.p_state_[0] = (self.yaw_ego_+1)/2 # jingda version: normalization
            # self.p_state_[0] = (0.576 * action[1]+1)/2 # jingda version: Sep verison
            self.p_state_[0] = (action[0])/2 # jingda version: Oct verison
            self.p_state_[1] = rho
            self.p_state_[2] = beta2

            # print('yaw_ego:{}'.format((0.576 * action[1]+1)/2))
            # print('rho:{}'.format(rho))
            # print('beta2:{}'.format(beta2))
            # print('pedal:{}'.format(action[0]))
            # print(self.p_state_)
    
            velMsg.linear.x = self.alpha_v_ *action[0]
            velMsg.angular.z = -0.576 * action[1]
            self.twistPublisher_.publish(velMsg)


    def inference(self):
        for i in range(3):
            self.img_state_[:,:,i] = self.image_deque_[i]
        with torch.no_grad():
            a = self.agent_.choose_action(self.img_state_, self.p_state_, evaluate=True)
        
        # print(a)s
        #jingda version
        # a[0]=np.clip(a[0],0,1)
        return a

    def takeover_listener_callback(self, triger_msg):
        self.takeover_deque_.append(triger_msg.data)

        if (triger_msg.data == 1):
            self.takeover_ = 1

        elif (triger_msg.data == 0):
            self.takeover_ = 0

        if((self.takeover_deque_[1] - self.takeover_deque_[0]) == -1):
            print("Reloading!")    
            self.agent_.policy.load_state_dict(torch.load('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/actor.pkl', map_location=torch.device('cpu')))
            self.agent_.critic.load_state_dict(torch.load('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/critic.pkl', map_location=torch.device('cpu')))
            print("Reloading finished!")

    def goal_callback(self, goal_msg):
        self.goal_.pose.pose.position.x = self.end_x_ #goal_msg.pose.position.x
        self.goal_.pose.pose.position.y = self.end_y_ #goal_msg.pose.position.y
        self.e2e_path_.poses = []

if __name__ == '__main__':
    print("System Initializing...")
    rospy.init_node('semantic_e2e_nav')
    e2eNavigation()
    rospy.spin()
