import re
import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semantic_segmentation.semseg.models import *
from semantic_segmentation.semseg.datasets import *
from semantic_segmentation.semseg.utils.utils import timer
from semantic_segmentation.semseg.utils.visualize import draw_text

from rich.console import Console

import rospy
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy

import numpy as np

console = Console()


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])
        self.seg_publisher_ = rospy.Publisher("/segnet/color_mask", Image, queue_size=10 )
        self.bridge_ = CvBridge()
        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        # if overlay: 
        #     seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        # image = draw_text(seg_image, seg_map, self.labels)
        return seg_image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, image: Tensor, overlay: bool) -> Tensor:
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        seg_map = np.uint8(seg_map.numpy())
        return seg_map

def segformer_callback(rgb_msg):
    try:
        rgb_img = ros_numpy.numpify(rgb_msg)[:,:,0:3]
        tmp = rgb_img
        # print(rgb_img.shape)
        rgb_img = torch.tensor(rgb_img).to(semseg.device).permute(2,0,1)
        # print(rgb_img.shape)
    except CvBridgeError as e:
        print(e)

    with console.status("[bright_green]Processing..."):
        segmap = semseg.predict(rgb_img, cfg['TEST']['OVERLAY'])
        seg_msg = Image()
        seg_msg = semseg.bridge_.cv2_to_imgmsg(segmap, encoding="rgb8")
        seg_msg.header.stamp = rospy.Time.now()

        semseg.seg_publisher_.publish(seg_msg)
        # segmap = segmap.detach().cpu().numpy()

        # print (segmap.shape)
        # added_image = cv2.addWeighted(tmp,0.4,segmap,1.0,0)
        # window_name = 'image'
        # cv2.imshow(window_name, added_image)
        # cv2.waitKey(10) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='semantic_segmentation/configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    semseg = SemSeg(cfg)

    rospy.init_node('segformer', anonymous=True)

    rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, segformer_callback)
    rospy.spin()


