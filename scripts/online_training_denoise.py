# ROS dependencies
#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy

import message_filters

# RL dependencies
import numpy as np
import torch

from SAC_based_DRL_denoise.sac import DRL

# OpenCV
import cv2

# Deque
from collections import deque

# File transfer
import paramiko
import time
import sys

class e2eOnline:
    def __init__(self):
        # DRL model initialization
        self.agent_ = DRL(action_dim=2, pstate_dim=3, BUFFER_SIZE=1e3)
        self.agent_.policy.load_state_dict(torch.load('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/actor10100.pkl', map_location=torch.device('cpu')))
        self.agent_.critic.load_state_dict(torch.load('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/critic10100.pkl', map_location=torch.device('cpu')))
        self.agent_.critic_target.load_state_dict(torch.load('/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/critic10100.pkl', map_location=torch.device('cpu')))

        # sync for joy and image state
        self.semantic_subscriber_ = message_filters.Subscriber("/segnet/color_mask", numpy_msg(Image))
        self.demo_joy_subscriber = message_filters.Subscriber("/joy", Joy)
        ts = message_filters.ApproximateTimeSynchronizer([self.semantic_subscriber_, self.demo_joy_subscriber], 100, 0.1, allow_headerless=False)
        ts.registerCallback(self.training_callback)

        self.demo_twist_publisher_ = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.img_publisher_ = rospy.Publisher("/automan/seg_img", Image, queue_size=10)
        # takeover signal: 1 means human guidence starts and 0 means demostration ends.        
        self.takeover_trigger_publisher_ = rospy.Publisher("/takeover", Int64, queue_size=10)
        self.joy_subscriber_ = rospy.Subscriber("/joy", Joy, self.status_signal_callback)

        self.bridge_ = CvBridge()

        self.init_ = False
        self.takeover_ = 0
        self.takeover_deque_ = deque(maxlen=2)
        self.takeover_end_deque_ = deque(maxlen=2)
        self.training_deque_ = deque(maxlen=2)

        for i in range(2):
            self.takeover_deque_.append(0)
            self.takeover_end_deque_.append(0)
            self.training_deque_.append(0)

        # colision and takeover shares the same trigger
        self.collision_ = 0
        self.training_ = 0
        
        # buffer for training part input
        self.image_deque_ = deque(maxlen = 3)
        self.image_state_deque_ = deque(maxlen=2)
        self.p_state_deque_ = deque(maxlen=2)
        self.counter_ = 1

        # training input 
        self.p_state_ = np.array([0, 0, 0], np.float32)
        self.img_state_ = np.repeat(np.expand_dims(np.zeros(shape=(90, 180)),2), 3, axis=2)
        
        # current state 
        self.velocity_ = 0
        self.yaw_ = 0

        print("System Initialized! Waiting for command!")

    def training_callback(self, semantic_msg, joy_msg):
        # image segmentation post processing
        # inflation for safety consideration
        try:
            img = np.float32(ros_numpy.numpify(semantic_msg))
            tmp = np.ones((img.shape[0], img.shape[1])).astype(np.float32)
            tmp[np.where((img[:,:,0]==0.0) & (img[:,:,1]==128.0) & (img[:,:,2]==0.0))]=0.05
            
            img1 = cv2.resize(tmp, (180, 90))
            img1 = cv2.medianBlur(img1, 5)
            kernel = np.ones((7, 7), np.uint8)
            semantic_img = cv2.dilate(img1, kernel) 
            semantic_img = cv2.GaussianBlur(semantic_img, (5,5), 0)
            
            # cv2.namedWindow("drivable area", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('drivable area', 640, 480) 
            # cv2.imshow('drivable area', semantic_img)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

        self.image_deque_.append(np.array(semantic_img))

        if (len(self.image_deque_)==3):
            self.init_ = True

        
        if (self.init_):
            # action = self.inference()
            # print(action)
            velMsg = Twist()
            self.velocity_ = np.double(joy_msg.axes[1])
            self.yaw_ = np.double(joy_msg.axes[2])
            # print(self.velocity_)
            # print(self.yaw_)

            # training input calculation
            
            for i in range(3):
                self.img_state_[:,:,i] = self.image_deque_[i]
            self.image_state_deque_.append(self.img_state_)

            self.p_state_[0] = self.velocity_
            self.p_state_[1] = self.yaw_
            self.p_state_[2] = self.velocity_

            self.p_state_deque_.append(self.p_state_)

            # if ((self.takeover_) & (not self.training_) & (self.counter_ < 65)):
            if ((self.takeover_) & (not self.training_)):
                # print("Demostatration starts! " + str(self.counter_))
                # when quene filled up, store the transition information 
                if ((len(self.image_state_deque_) == 2) & (len(self.p_state_deque_) == 2)):
                    print("Demostatration starts! " + str(self.counter_))
                    reward = self.reward_calculation()
                    self.agent_.store_transition(self.image_state_deque_[0], self.p_state_deque_[0], self.p_state_[0:2], self.p_state_[0:2], 1, reward, self.image_state_deque_[1], self.p_state_deque_[1], self.collision_)
                    self.counter_+=1

                velMsg.linear.x = self.velocity_
                velMsg.angular.z = self.yaw_

    
                
            elif((self.takeover_) & (self.training_)):
                velMsg.linear.x = 0
                velMsg.angular.z = 0
            
            else:
                return
            
            # print("Ready for sending command!")
            self.demo_twist_publisher_.publish(velMsg)
            
            


    def online_training(self):
        print("The online training part starts!")
        for _ in range(5):
            self.agent_.learn(64)
        torch.save(self.agent_.policy.state_dict(), '/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/actor.pkl')
        torch.save(self.agent_.critic.state_dict(), '/home/jingda/Workspace/catkin_ws/src/e2e_nav/scripts/models/critic.pkl')
        print("The new model has already finished!")

    def reward_calculation(self):
        reward = 0.1 - 20 * (self.collision_)
        reward += 0.2 * (self.velocity_)
        reward -= 1 * abs(self.yaw_)
        # reward -= 5 * self.freeze
        return reward

    def model_upload(self, src, dst):
        print("The new model file is uploading to the AGV!")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect('localhost', username='jingda', password=' ', timeout=5)
            print("Connected successfully!")
        except:
            print("Connection failed!")
            sys.exit()

        sftp = ssh.open_sftp() 
        timeout = time.time() + 5
        while True:
            try:
                sftp.put(src, dst)
                print ("Copied successfully!")
                break
            except:
                print("Oooooops! Try to re-send!")
                if (time.time() > timeout):
                    print("File transferring fails! Please training model online again!")
                    break
            time.sleep(1)

        sftp.close()
        print("The file has already arrived, please finish the online training!")
    
    # Record takeover status
    def status_signal_callback(self, joy_msg):
        takeover_msg = Int64()
        self.collision_ = 0
        self.takeover_deque_.append(joy_msg.buttons[0])
        self.takeover_end_deque_.append(joy_msg.buttons[2])
        self.training_deque_.append(joy_msg.buttons[1])

        if ((self.takeover_deque_[1] - self.takeover_deque_[0]) == 1):
            self.collision_ = 1
            self.takeover_ = 1
            print("Takeover starts!")
        
        elif ((self.training_deque_[1] - self.training_deque_[0]) == 1):
            self.training_ = 1
            self.online_training()
            # self.agent_.replay_buffer.clear()
            # print(self.agent_.replay_buffer.get_current.episode_len())
            # self.model_upload()
            # src_actor = '/home/jingda/Workspace/catkin_ws/src/e2e_nav_online/scripts/models/actor.pkl'
            # dst_actor = '/home/scoutmini/Workspace/ws_automan/src/e2e_nav/scripts/models/actor.pkl'
            # self.model_upload(src_actor, dst_actor)

            # src_critic = '/home/jingda/Workspace/catkin_ws/src/e2e_nav_online/scripts/models/critic.pkl'
            # dst_critic = '/home/scoutmini/Workspace/ws_automan/src/e2e_nav/scripts/models/critic.pkl'
            # self.model_upload(src_critic, dst_critic)

            # print('eee')
        elif ((self.takeover_end_deque_[1] - self.takeover_end_deque_[0]) == 1):
            self.takeover_ = 0
            self.training_ = 0
            self.counter_ = 0
            print("Takeover ends!")

        takeover_msg.data = self.takeover_
        self.takeover_trigger_publisher_.publish(takeover_msg)


if __name__ == '__main__':
    print("System Initializing...")
    rospy.init_node('e2e_online_training')
    e2eOnline()
    rospy.spin()
