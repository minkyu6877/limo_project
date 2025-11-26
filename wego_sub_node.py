#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import cv2
import numpy as np
from cv_bridge import CvBridge
from time import time

class Class_sub:
    def __init__(self) :
        rospy.init_node("wego_sub_node") 
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        self.camera_msg = CompressedImage() 
        self.cmd_msg = Twist() 
        self.bridge = CvBridge()
        
        # 제어 주기 20Hz (안정적)
        self.rate = rospy.Rate(20) 
        
        self.camera_flag = False
        self.start = True

        #------------------Lidar-------------------------#
        self.msg = None
        self.is_scan = False 
        self.obstacle_flag = False # 장애물 감지 여부

        # [수정] 장애물 감지 설정 (확실하게 감지하도록)
        self.scan_Ldgree = 45   # 좌우 45도 (총 90도) 감시
        self.scan_Rdgree = 45   
        self.min_dist = 0.7     # 0.7m 이내 들어오면 무조건 회피 시작

        # [수정] 속도 설정 (안전하게)
        self.default_speed = 0.12 
        self.camera_speed = 0.12 
        
        self.speed = 0
        self.angle = 0
        self.steer = 0

        #------------------Flag-------------------------#
        self.v2x_flag = False
        self.v2x = "D"

        # [HSV] 노란색/흰색/검은색 (현장 조명에 맞춰야 함)
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 60])
        self.yellow_lower = np.array([10, 50, 50])
        self.yellow_upper = np.array([40, 255, 255])
        
        self.steer_weight = 2.0 

        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/path_", String, self.v2x_cb)

    def lidar_cb(self, msg): self.msg = msg 
    def camera_cb(self, msg):
        if msg is not None:
            self.camera_msg = msg
            self.camera_flag = True
        else: self.camera_flag = False
    def v2x_cb(self, msg):
        if msg is not None: self.v2x = msg.data

    # ================= [핵심 수정] 장애물 감지 로직 ================= #
    def check_obstacle(self):
        if self.msg is None: return False
        
        # 전방 90도(-45~45) 영역의 거리 데이터 확인
        ranges = np.array(self.msg.ranges)
        degrees = [(self.msg.angle_min + i * self.msg.angle_increment) * 180/math.pi for i in range(len(ranges))]
        
        obstacle_cnt = 0
        for i, dist in enumerate(ranges):
            if -self.scan_Rdgree < degrees[i] < self.scan_Ldgree:
                # 0.1m(너무 가까운 노이즈) ~ 0.7m(감지거리) 사이 장애물 카운트
                if 0.1 < dist < self.min_dist:
                    obstacle_cnt += 1
        
        # 점이 5개 이상 감지되면 장애물로 판단
        if obstacle_cnt > 5:
            return True
        else:
            return False

    # ================= [핵심 수정] 장애물 회피 로직 (Gap Following) ================= #
    def avoid_obstacle(self):
        if self.msg is None: return

        ranges = np.array(self.msg.ranges)
        degrees = [(self.msg.angle_min + i * self.msg.angle_increment) * 180/math.pi for i in range(len(ranges))]
        
        left_space = 0
        right_space = 0
        
        # 왼쪽(-45~0도)과 오른쪽(0~45도) 중 어디가 더 넓은지 비교
        # (거리가 멀수록 점수가 높음)
        for i, dist in enumerate(ranges):
            angle = degrees[i]
            if 0 < dist < 2.0: # 유효 거리 2m 이내만 계산
                if 0 < angle < self.scan_Ldgree: # 왼쪽
                    left_space += dist
                elif -self.scan_Rdgree < angle < 0: # 오른쪽
                    right_space += dist
        
        # 더 넓은 쪽으로 핸들 꺾기
        if left_space > right_space:
            self.angle = 0.5  # 좌회전 (약 30도)
        else:
            self.angle = -0.5 # 우회전
            
        self.speed = self.default_speed # 속도 유지 (멈추지 않게)

    # ================= 카메라 처리 (기존 최적화 유지) ================= #
    def lkas(self):
        if not self.camera_flag: return

        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(self.camera_msg)
            cv_img = cv2.resize(cv_img, (320, 240)) # 리사이즈 (속도 향상)
            h, w = cv_img.shape[:2]
            hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

            # 차선 검출 (검정색 or 노란색 or 흰색 - 필요에 따라 수정)
            # 지금은 검정색(도로) 인식으로 되어 있음
            mask = cv2.inRange(hsv_img, self.black_lower, self.black_upper)
            
            # ROI (하단부만)
            mask_roi = mask[int(h*0.6):, :] 
            
            # 무게 중심 찾기 (가장 단순하고 빠른 방법)
            M = cv2.moments(mask_roi)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                err = (w // 2) - cx
                self.steer = (err * math.pi / w) * self.steer_weight
            else:
                self.steer = 0
        except:
            pass

    # ================= 메인 제어 ================= #
    def ctrl(self):
        # 1. 카메라 영상 처리 (항상 수행)
        self.lkas()

        # 2. 장애물 확인
        self.obstacle_flag = self.check_obstacle()

        if self.obstacle_flag:
            # 장애물 있으면 -> 라이다 회피 주행
            self.avoid_obstacle()
            self.cmd_msg.linear.x = self.speed
            self.cmd_msg.angular.z = self.angle
            # print("Avoid Mode!")
        else:
            # 장애물 없으면 -> 카메라 라인트레이싱
            self.cmd_msg.linear.x = self.camera_speed
            self.cmd_msg.angular.z = self.steer
            # print("Line Mode")

        self.pub.publish(self.cmd_msg)
        self.rate.sleep()

if __name__ == "__main__":
    class_sub = Class_sub()
    
    try:
        input("Press Enter to Start...")
    except:
        pass

    while not rospy.is_shutdown():
        class_sub.ctrl()
