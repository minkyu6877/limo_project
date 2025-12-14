#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import cv2
import numpy as np
from cv_bridge import CvBridge
from time import time, sleep

class Class_sub:
    def __init__(self):
        rospy.init_node("wego_sub_node")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        self.camera_msg = CompressedImage() 
        self.cmd_msg = Twist() 
        self.bridge = CvBridge()
        
        self.rate = rospy.Rate(20) 
        
        self.camera_flag = False
        self.start = True

        #------------------LiDAR 설정-------------------------#
        self.scan_ranges = []
        self.front = 999.0

        #------------------상태 변수-------------------------#
        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start_time = 0.0

        # [속도 설정]
        self.speed_fwd = 0.15       
        self.speed_back = -0.15     
        self.camera_speed = 0.15     

        #------------------LKAS 변수-------------------------#
        self.steer_weight = 3.5
        self.steer = 0.0

        # Flag & Timer
        self.flag6 = False       
        self.v2x_flag = False
        self.mission_ABC = False 
        self.yellow_long_detection = False
        self.flag6_count = 0 
        self.prev = False
        self.current = False
        self.count = 0
        self.ABC_time = 0
        self.v2x = "D"

        # HSV Setting
        self.black_lower = np.array([102, 0, 60])
        self.black_upper = np.array([164, 86, 136])
        self.black2_lower = np.array([126, 25, 45])
        self.black2_upper = np.array([167, 89, 108])
        self.black3_lower = np.array([125, 29, 26])
        self.black3_upper = np.array([171, 100, 78])
        self.yellow_lower = np.array([14, 17, 153])
        self.yellow_upper = np.array([35, 167, 255])
        
        self.margin_x = 150
        self.margin_y = 350

        rospy.Subscriber("/scan", LaserScan, self.lidar_cb) 
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/path_", String, self.v2x_cb)

    #------------------LiDAR Callback-------------------------#
    def lidar_cb(self, scan): 
        self.scan_ranges = np.array(scan.ranges)
        
        # 정면 감지 (장애물 인식용) - 30도
        raw = self.scan_ranges
        front_zone = np.concatenate([raw[:30], raw[-30:]])
        cleaned = [d for d in front_zone if d != 0 and not np.isinf(d) and d > 0.08]

        if len(cleaned) > 0:
            self.front = float(min(cleaned))
        else:
            self.front = 999.0

    def camera_cb(self, msg):  
        if msg != None:
            self.camera_msg = msg
            self.camera_flag = True
        else :
            self.camera_flag = False

    def v2x_cb(self, msg):
        if msg != None:
            self.v2x = msg.data

    #------------------Mission 4 Logic (중앙 가중치 + 광각)-------------------------#
    def find_gap_once(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        
        # [수정 1] 탐색 각도 복구 (±45 -> ±75)
        # 다시 넓게 봅니다. 옆에 있는 틈새도 놓치지 않기 위함입니다.
        scan_angle = 75
        ranges = np.concatenate([raw[-scan_angle:], raw[:scan_angle]])
        ranges = np.where((ranges < 0.08) | np.isnan(ranges) | np.isinf(ranges), 0.0, ranges)

        gaps = []
        start = None

        for i, d in enumerate(ranges):
            # 20cm 이상 공간 탐색
            if d > 0.20:
                if start is None:
                    start = i
            else:
                if start is not None:
                    gaps.append((start, i - 1))
                    start = None

        if start is not None:
            gaps.append((start, len(ranges) - 1))

        if not gaps:
            return 0.0

        # [수정 2] 중앙 가산점(Score) 로직 적용
        best_gap = None
        max_score = -99999
        center_index = scan_angle # 75 (배열의 중앙 인덱스)

        for g in gaps:
            # 1. 갭의 너비 (클수록 좋음)
            width = g[1] - g[0]
            
            # 2. 갭의 위치 (중앙 인덱스 75와 얼마나 먼가?)
            gap_center = (g[0] + g[1]) / 2
            distance_from_center = abs(gap_center - center_index)
            
            # [핵심] 점수 계산 공식
            # Score = 너비 - (중앙과의 거리 * 가중치)
            score = width - (distance_from_center * 1.5)

            # 너무 작은 갭(노이즈)은 무시
            if width > 5: 
                if score > max_score:
                    max_score = score
                    best_gap = g
        
        # 만약 점수 계산으로 선택된 게 없으면 그냥 가장 큰 것 선택 (안전장치)
        if best_gap is None:
            best_gap = max(gaps, key=lambda g: g[1] - g[0])

        mid = (best_gap[0] + best_gap[1]) // 2

        # 75도 기준 인덱스 변환
        angle = (mid - scan_angle) * (np.pi / 180.0)
        
        return angle

    def run_avoidance_logic(self):
        current_time = time()
        
        # [왼쪽 벽 방어 로직 유지]
        if len(self.scan_ranges) > 60:
            left_check = self.scan_ranges[10:60]
            valid_left = left_check[(left_check > 0.05) & (left_check < 1.0)]
            if len(valid_left) > 0 and np.min(valid_left) < 0.25:
                # print(f"!!! LEFT WALL TOO CLOSE -> FORCE RIGHT !!!")
                self.cmd_msg.linear.x = 0.1  
                self.cmd_msg.angular.z = -0.6 
                self.pub.publish(self.cmd_msg) 
                return 

        if self.state == "BACK":
            if current_time - self.state_start_time < 1.8:
                self.cmd_msg.linear.x = self.speed_back
                self.cmd_msg.angular.z = 0.0
            else:
                self.escape_angle = self.find_gap_once()
                self.state = "ESCAPE"
                self.state_start_time = current_time

        elif self.state == "ESCAPE":
            if current_time - self.state_start_time < 2.2:
                self.cmd_msg.linear.x = 0.15 
                
                target_steer = self.escape_angle * 1.0
                if target_steer > 0.5: target_steer = 0.5
                if target_steer < -0.5: target_steer = -0.5
                
                self.cmd_msg.angular.z = target_steer
            else:
                print("ESCAPE Done -> Return to LANE")
                self.state = "LANE"
                self.state_start_time = current_time

    #----------------------------------LKAS------------------------------------#
    def lkas(self):
        if self.camera_flag == True:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(self.camera_msg, "bgr8")
            y, x, channel = cv_img.shape
            hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

            yellow_filter = cv2.inRange(hsv_img, self.yellow_lower, self.yellow_upper)
            roi_mask = np.zeros_like(yellow_filter) 
            roi_mask[200:y, 0:x] = 255 
            yellow_filter_roi = cv2.bitwise_and(yellow_filter, roi_mask)
            yellow_pixel = cv2.countNonZero(yellow_filter_roi)
            
            black_filter = cv2.inRange(hsv_img, self.black_lower, self.black_upper)
            black2_filter = cv2.inRange(hsv_img, self.black2_lower, self.black2_upper)
            black3_filter = cv2.inRange(hsv_img, self.black3_lower, self.black3_upper)
            filter = cv2.bitwise_or(black_filter, black2_filter)
            combine_filter = cv2.bitwise_or(filter, black3_filter)
            and_img = cv2.bitwise_and(cv_img, cv_img, mask=combine_filter)

            if yellow_pixel > 7000:
                self.yellow_long_detection = True
            
            if self.yellow_long_detection:
                if yellow_pixel > 1500: self.count += 1
                else: self.count = 0
        
            if self.count > 30: 
                self.current = True
                self.prev = True
            else:
                self.current = False

            if self.prev and not self.current and self.flag6_count == 0:
                self.flag6 = True
                self.flag6_count = 1
                self.prev = False

            src_pts = np.float32([(30, y), (self.margin_x, self.margin_y), 
                                  (x - self.margin_x, self.margin_y), (x - 30, y)])
            dst_margin_x = 120
            dst_pts = np.float32([(dst_margin_x, y), (dst_margin_x, 0), 
                                  (x - dst_margin_x, 0), (x - dst_margin_x, y)])

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp_img = cv2.warpPerspective(and_img, matrix, (x, y))
            gray_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
            bin_img = np.zeros_like(gray_img)
            bin_img[gray_img != 0] = 1
            center_index = x // 2

            window_num = 8
            window_y_size = y // window_num 
            left_indices = []
            right_indices = []

            for i in range(window_num):
                upper_y = y - window_y_size * (i + 1)
                lower_y = y - window_y_size * i
                
                left_window = bin_img[upper_y:lower_y, :center_index]
                left_hist = np.sum(left_window, axis=0)
                left_hist[left_hist < 40] = 0

                right_window = bin_img[upper_y:lower_y, center_index:]
                right_hist = np.sum(right_window, axis=0)
                right_hist[right_hist < 40] = 0

                try:
                    l_nz = np.nonzero(left_hist)[0]
                    r_nz = np.nonzero(right_hist)[0]
                    if len(l_nz) > 0:
                        left_indices.append((l_nz[0] + l_nz[-1]) // 2)
                    if len(r_nz) > 0:
                        right_indices.append((r_nz[0] + r_nz[-1]) // 2 + center_index)
                except: pass
            
            try:
                if left_indices and right_indices:
                    l_avg = sum(left_indices) / len(left_indices)
                    r_avg = sum(right_indices) / len(right_indices)
                    avg_idx = int((l_avg + r_avg) // 2)
                    error = center_index - avg_idx
                    self.steer = (error * math.pi / x) * self.steer_weight
            except: pass

    #---------------------------MAIN CONTROL------------------------------------#
    def ctrl(self):
        self.lkas() 

        if self.state == "LANE":
            print(f"Front Dist: {self.front:.2f}")

            if self.front < 0.30:
                print("Cone Detected! -> Switch to BACK")
                self.state = "BACK"
                self.state_start_time = time()
                self.cmd_msg.linear.x = 0
                self.cmd_msg.angular.z = 0
            else:
                self.cmd_msg.linear.x = self.camera_speed
                self.cmd_msg.angular.z = self.steer
        
        else:
            self.run_avoidance_logic()
        
        if self.flag6:
            self.v2x_flag = True
            self.flag6 = False 

        if self.v2x_flag: 
            if self.v2x == "D":
                self.cmd_msg.linear.x = 0
                self.cmd_msg.angular.z = 0
            else :
                self.camera_speed = 0.12
                self.v2x_flag = False 
                self.mission_ABC = True
                self.ABC_time = time()

        if self.mission_ABC:
            current_time = time()
            diff = current_time - self.ABC_time
            if 0 < diff < 2:
                if self.v2x == "A": self.cmd_msg.angular.z = 0.55
            elif 2 < diff < 4:
                if self.v2x == "B": self.cmd_msg.angular.z = 0.25 
            elif 6 < diff < 7:
                if self.v2x == "C": self.cmd_msg.angular.z = -0.15 
            elif diff > 7:
                self.mission_ABC = False
        
        self.pub.publish(self.cmd_msg)
        self.rate.sleep()


if __name__ == "__main__":
    class_sub = Class_sub()
    
    print("Ready to go... Starting in 3 seconds!")
    sleep(3)
    print("GO!")

    while not rospy.is_shutdown():
        class_sub.ctrl()

    stop_msg = Twist()
    stop_msg.linear.x = 0
    stop_msg.angular.z = 0
    class_sub.pub.publish(stop_msg)

    rospy.signal_shutdown("Finished")
