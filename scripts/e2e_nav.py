# ROS dependencies
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy
# RL dependencies
import numpy as np
import torch
from SAC_based_DRL_denoise.sac import DRL
import cv2

#Deque
from collections import deque

class e2eNavigation:
    def __init__(self):
        self.agent_ = DRL(action_dim=2, pstate_dim=3, BUFFER_SIZE=1e0)
        self.agent_.policy.load_state_dict(torch.load('/home/scoutmini/Workspace/ws_automan/src/e2e_nav/scripts/models/actor.pkl', map_location=torch.device('cpu')))
        self.agent_.critic.load_state_dict(torch.load('/home/scoutmini/Workspace/ws_automan/src/e2e_nav/scripts/models/critic_denoise1.pkl', map_location=torch.device('cpu')))
        self.semantic_subscriber_ = rospy.Subscriber("/segnet/color_mask", numpy_msg(Image), self.semantic_callback)
        self.takeover_trigger_subscriber = rospy.Subscriber("/takeover", Int64, self.takeover_listener_callback)

        self.twistPublisher_ = rospy.Publisher("/cmd_vel", Twist, queue_size=30)
        self.img_publisher_ = rospy.Publisher("/automan/seg_img", Image, queue_size=10 )
        self.bridge_ = CvBridge()

        self.init_ = False
        self.takeover_ = 0
        self.takeover_deque_ = deque(maxlen = 2)
        for i in range(2):
            self.takeover_deque_.append(0)

        self.image_deque_ = deque(maxlen = 3)

        self.p_state_ = np.array([0, 0, 0])
        self.img_state_ = np.repeat(np.expand_dims(np.zeros(shape=(90, 180)),2), 3, axis=2)
        self.alpha_v_ = 0.4
        print("System Initialized! When drivable area imge showing , the inference starts!")

    def semantic_callback(self, semantic_msg):
        try:
            img = np.float32(ros_numpy.numpify(semantic_msg))
            tmp = np.ones((img.shape[0], img.shape[1])).astype(np.float32)
            tmp[np.where((img[:,:,0]==0.0) & (img[:,:,1]==128.0) & (img[:,:,2]==0.0))]=0.05
            
            img1 = cv2.resize(tmp, (180, 90))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(img, (128, 64))
            # img1 = np.ones_like(img)
            # img1 = img1.astype(np.float32)
            # img1[np.where((img>72) & (img<77))]=0.05
            img1 = cv2.medianBlur(img1, 5)
            kernel = np.ones((7, 7), np.uint8)
            semantic_img = cv2.dilate(img1, kernel) 
            semantic_img = cv2.GaussianBlur(semantic_img, (5,5), 0)
            
            #cv2.namedWindow("drivable area", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('drivable area', 640, 480) 
            #cv2.imshow('drivable area', semantic_img)
            #cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

        self.image_deque_.append(np.array(semantic_img))

        if len(self.image_deque_)==3:
            self.init_ = True

        velMsg = Twist()

        if (self.init_ & (not self.takeover_)):
            action = self.inference()
            #print(action)
            self.p_state_[0] = self.alpha_v_ *action[0]
            self.p_state_[1] = -0.576 *action[1]
            self.p_state_[2] = self.alpha_v_ *action[0]

            velMsg.linear.x = self.alpha_v_ *action[0] #* 0.5
            velMsg.angular.z = -0.576 * action[1]
            self.twistPublisher_.publish(velMsg)

        self.img_publisher_.publish(self.bridge_.cv2_to_imgmsg(semantic_img))

    def inference(self):
        for i in range(3):
            self.img_state_[:,:,i] = self.image_deque_[i]
        with torch.no_grad():
            a = self.agent_.choose_action(self.img_state_, self.p_state_, evaluate=True)
        a[0] = np.clip(a[0],0,1)
        return a

    def takeover_listener_callback(self, triger_msg):
        self.takeover_deque_.append(triger_msg.data)

        if (triger_msg.data == 1):
            self.takeover_ = 1

        elif (triger_msg.data == 0):
            self.takeover_ = 0

        if((self.takeover_deque_[1] - self.takeover_deque_[0]) == -1):
            print("Reloading!")    
            self.agent_.policy.load_state_dict(torch.load('/home/scoutmini/Workspace/ws_automan/src/e2e_nav/scripts/models/actor.pkl', map_location=torch.device('cpu')))
            self.agent_.critic.load_state_dict(torch.load('/home/scoutmini/Workspace/ws_automan/src/e2e_nav/scripts/models/critic.pkl', map_location=torch.device('cpu')))
            print("Reloading finished!")

if __name__ == '__main__':
    print("System Initializing...")
    rospy.init_node('semantic_e2e_nav')
    e2eNavigation()
    rospy.spin()
