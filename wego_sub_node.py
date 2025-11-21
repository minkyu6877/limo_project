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
        rospy.init_node("wego_sub_node") # 1. node 이름 설정
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        # self.lidar_msg = LaserScan()
        self.camera_msg = CompressedImage() # for publish 
        self.cmd_msg = Twist() 
        self.bridge = CvBridge()
        
        # [수정 1] 루프 주기 45Hz -> 20Hz (연산 부하 감소 및 안정화)
        self.rate = rospy.Rate(20) 
        
        self.camera_flag = False
        self.e_stop_flag = False
        self.obstacle_flag = False
        self.steer = 0
        self.start = True

        #------------------Lidar-------------------------#
        self.msg = None
        self.lidar_flag = False
        self.dist_data = 0
        self.direction = None

        self.is_scan = False # 메인함수에서 라이더 함수실행 

        # LIDAR test
        # self.is_scan = True
    
        self.obstacle_ranges = []
        self.center_list_left = []
        self.center_list_right = []

        self.scan_Ldgree = 25   # decrease corn 왼쪽 스캔 각 
        self.scan_Rdgree = 25   # 오른쪽 스캔 각
        self.min_dist = 0.5

        self.speed = 0
        self.angle = 0
        self.default_speed = 0.15
        self.default_angle = 0.0
        self.turning_speed = 0.08
        self.backward_speed = -0.08
        self.OBSTACLE_PERCEPTION_BOUNDARY = 8     # decrease corn 조건문에서 낮추면 장애물 많이 감지 
        self.ranges_length = None
        self.obstacle_exit = False # 초기화 추가

        #----------------------flag-----------------------# 
        self.flag4 = False      # 라바콘 감지 flag
        self.flag6 = False      # 노랑색 선 끝나는거 감지 
        self.flag6_count = 0    # 노랑색 두번 인식 방지 
        self.v2x_flag = False
        self.prev = False       # 처음 노랑색 감지 시점 
        self.current = False    # 노랑색 끝나는 지점 감지 

        self.mission_ABC = False # 미션6 시작 시점
        self.prev_red = False    # 빨강색을 봤을 때 true로 바뀜
        self.current_red = False # 현재 빨강색이 인지 됐는지 확인 
        self.yellow_long_detection = False  # 노랑색을 길게 감지하는 시점  

        #---------------------시간관련변수---------------------#
        self.ABC_time = 0  # 노랑색 선 끊기는 지점부터 시작 시간 
        self.mission3_count = 0  # 미션3에서 쓰이는 시간 
        self.count = 0           # 노랑색 얼마나 인지 했는지 시간 확인 (길게 인지하면 노랑 차선을 인지한것임)

        #-------------------------V2X-------------------------#
        self.v2x = "D"

        # FPS 계산을 위한 변수 초기화
        self.frame_count = 0  
        self.last_time = time() 
        self.fps = 0  

        #-------------------------HSV-------------------------#

        # black
        self.black_lower = np.array([102, 0, 60])
        self.black_upper = np.array([164, 86, 136])

        self.black2_lower = np.array([126, 25, 45])
        self.black2_upper = np.array([167, 89, 108])

        self.black3_lower = np.array([125, 29, 26])
        self.black3_upper = np.array([171, 100, 78])

        # yellow
        self.yellow_lower = np.array([14, 17, 153])
        self.yellow_upper = np.array([35, 167, 255])

        # red
        self.red_lower = np.array([167, 80, 115])
        self.red_upper = np.array([179, 205, 255])

        #-------------------------ROI-------------------------#

        self.margin_x = 150
        self.margin_y = 350

        self.camera_speed = 0.3
        self.steer_weight = 2.5
        
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb) # 2. node 역할 설정
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb) # 2. node 역할 설정
        rospy.Subscriber("/path_", String, self.v2x_cb)


    #-------------------------E M E R G E N C Y-------------------------------#
    def emergency(self): 
        # flag6 one more time 
        self.flag6_count = 0
        # no more mission3 
        self.mission3_count = 0
        self.is_scan = True
        self.camera_speed = 0.15
        self.steer_weight = 1.7
        self.start = True # 타이머 리셋을 위해 필요할 수 있음
        print ("EMERGENCY ACTIVE")

    #-------------------------C A L L B A C K-------------------------------#
    def lidar_cb(self, msg): # 3. subscriber - callback
        self.msg = msg # lidar
        if len(self.obstacle_ranges) > self.OBSTACLE_PERCEPTION_BOUNDARY:
            self.obstacle_exit = True
        else:
            self.obstacle_exit = False
        
        # self.is_scan = True  # 라이다 함수 바로실행
        self.obstacle_ranges = []

    def camera_cb(self, msg):  
        if msg != None:
            self.camera_msg = msg
            self.camera_flag = True
        else :
            self.camera_flag = False

    def v2x_cb(self, msg):
        if msg != None:
            # print(self.v2x)
            self.v2x = msg.data
            print(f"V2X Signal Received: {msg.data}")
            
    #--------------------------------L i D A R----------------------------------#     
    def LiDAR_scan(self):
        # [수정 2] msg가 None일 때의 예외처리 (안전장치)
        if self.msg is None:
            return 0, 0, 0, 0

        obstacle = []
        if self.lidar_flag == False:
            self.degrees = [
                (self.msg.angle_min + (i * self.msg.angle_increment)) * 180 / math.pi
                for i, data in enumerate(self.msg.ranges)
            ]
            self.ranges_length = len(self.msg.ranges)
            self.lidar_flag = True

        for i, data in enumerate(self.msg.ranges):
            if 0 < data < 0.5 and -self.scan_Rdgree < self.degrees[i] < self.scan_Ldgree:
                obstacle.append(i)
                self.dist_data = data

        if obstacle:
            self.obstacle_flag = True
            first = obstacle[0]
            first_dst = first
            last = obstacle[-1]
            last_dst = self.ranges_length - last
            self.obstacle_ranges = self.msg.ranges[first: last + 1]
        else:
            self.obstacle_flag = False
            first, first_dst, last, last_dst = 0, 0, 0, 0
        
        return first, first_dst, last, last_dst
    
    def move_direction(self, last, first):
        if self.direction == "right":
            for i in range(first):
                self.center_list_left.append(i)
            # [수정 3] math.floor 사용
            if self.center_list_left:
                Lcenter = self.center_list_left[math.floor(first/2)]
                center_angle_left = -self.msg.angle_increment * Lcenter
                self.angle = center_angle_left/3.0
                self.speed = self.default_speed * 0.9

        elif self.direction == "left":
            for i in range(len(self.msg.ranges)):
                self.center_list_right.append(last+i)
            # [수정 3] math.floor 사용 및 index error 방지
            if self.center_list_right:
                Rcenter = self.center_list_right[
                    math.floor(last + (self.ranges_length - last)/2)
                ]
                center_angle_right = self.msg.angle_increment * Rcenter
                self.angle = center_angle_right/4.0  
                self.speed = self.default_speed * 0.9

        elif self.direction == "back":
            self.angle = self.default_angle # 0
            self.speed = self.backward_speed # -0.08

        else:
            self.angle = self.default_angle
            self.speed = self.default_speed

    def compare_space(self, first_dst, last_dst):
        if self.obstacle_exit == True:
            if first_dst > last_dst: 
                # print("right")
                self.direction = "right"
            elif first_dst <= last_dst: 
                self.direction = "left"
                # print("left")
            else:
                self.direction = "back"
                print("back")
        else:
            self.direction = "front"

    #----------------------------------C A M E R A------------------------------------#
    def lkas(self):
        if self.camera_flag == True:
            
            cv_img = self.bridge.compressed_imgmsg_to_cv2(self.camera_msg)
            y, x, channel = cv_img.shape
            hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

            red_filter = cv2.inRange(hsv_img, self.red_lower, self.red_upper)
            roi_mask = np.zeros_like(red_filter)
            roi_mask[200:y, 0:x] = 255
            red_filter_roi = cv2.bitwise_and(red_filter, roi_mask)
            red_pixel = cv2.countNonZero(red_filter_roi)

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

            #--------------------------FLAG-----------------------------------#
            #================flag4===============#
            if red_pixel > 30000 and self.prev_red == False:
                self.flag4 = True
                self.current_red = True
                self.prev_red = True
            elif red_pixel < 2500:
                 self.current_red = False
            
            if not self.current_red and self.prev_red:
                print("----------------mission5-----------------")
                self.prev_red = False

            #================flag6===============#
            if yellow_pixel > 7000:
                self.yellow_long_detection = True
            
            if self.yellow_long_detection:
                if yellow_pixel > 1500:
                    self.count = self.count + 1
                else: 
                    self.count = 0
        
            if self.count > 30: # yellowlane
                self.current = True
                self.prev = True
            else:
                self.current = False

            if self.prev and not self.current and self.flag6_count == 0:  # True → False
                self.flag6 = True
                self.flag6_count = 1
                self.prev = False

            #---------------------------------------------------------------#

            src_pt1 = (30, y)
            src_pt2 = (self.margin_x, self.margin_y) ###
            src_pt3 = (x - self.margin_x, self.margin_y) ###
            src_pt4 = (x - 30, y)
            src_pts = np.float32([src_pt1, src_pt2, src_pt3, src_pt4])

            dst_margin_x = 120

            dst_pt1 = (dst_margin_x, y)
            dst_pt2 = (dst_margin_x, 0)
            dst_pt3 = (x - dst_margin_x, 0)
            dst_pt4 = (x - dst_margin_x, y)
            dst_pts = np.float32([dst_pt1, dst_pt2, dst_pt3, dst_pt4])

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            matrix_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            warp_img = cv2.warpPerspective(and_img, matrix, (x, y))
            gray_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
            bin_img = np.zeros_like(gray_img)
            bin_img[gray_img != 0] = 1
            center_index = x // 2

            window_num = 8
            margin = 40
            window_y_size = y // window_num # 60
            left_indices = []
            right_indices = []

            for i in range(0, window_num):
                upper_y = y - window_y_size * (i + 1)
                lower_y = y - window_y_size * i

                left_window = bin_img[upper_y:lower_y, :center_index]
                left_histogram = np.sum(left_window, axis=0)
                left_histogram[left_histogram < 40] = 0

                right_window = bin_img[upper_y:lower_y, center_index:]
                right_histogram = np.sum(right_window, axis=0)
                right_histogram[right_histogram < 40] = 0

                try:
                    left_nonzero = np.nonzero(left_histogram)[0]
                    left_avg_index = (left_nonzero[0] + left_nonzero[-1]) // 2
                    left_indices.append(left_avg_index)

                    right_nonzero = np.nonzero(right_histogram)[0]
                    right_avg_index = (right_nonzero[0] + right_nonzero[-1]) // 2 + center_index
                    right_indices.append(right_avg_index)

                    cv2.line(warp_img, (left_avg_index, y - window_y_size * (i + 1) + window_y_size // 2), (left_avg_index, y - window_y_size * (i + 1) + window_y_size // 2), (0, 0, 255), 10)
                    cv2.line(warp_img, (right_avg_index, y - window_y_size * (i + 1) + window_y_size // 2), (right_avg_index, y - window_y_size * (i + 1) + window_y_size // 2), (255, 0, 0), 10)
                    cv2.rectangle(warp_img, (left_avg_index - margin, upper_y), (left_avg_index + margin, lower_y), (255, 0, 0), 3)
                    cv2.rectangle(warp_img, (right_avg_index - margin, upper_y), (right_avg_index + margin, lower_y), (0, 0, 255), 3)
                    
                    # [수정 4] np.average 대신 단순 변수 사용 권장 (형변환 오류 방지)
                    # left_avg_indices = np.average(left_indices)
                    # right_avg_indices = np.average(right_indices)
                except :
                    pass
            
            # Steering Control Logic
            try:
                # 안전한 평균 계산
                if left_indices and right_indices:
                    left_avg = sum(left_indices) / len(left_indices)
                    right_avg = sum(right_indices) / len(right_indices)
                    avg_indices = int((left_avg + right_avg) // 2)

                    cv2.line(warp_img, (avg_indices, 0), (avg_indices, y), (0, 255, 255), 3)
                    error_index = center_index - avg_indices
                    # [수정 5] math.pi 사용
                    self.steer = (error_index * math.pi / x) * self.steer_weight
                else:
                    # 차선을 못 찾았을 경우 조향 유지 혹은 0 (상황에 따라 주석 해제)
                    # self.steer = 0 
                    pass

            except:
                pass
            
            cv2.line(warp_img, (center_index, 0), (center_index, y), (0, 255, 0), 3)
            # warp_inv_img = cv2.warpPerspective(warp_img, matrix_inv, (x, y))

            cv2.line(cv_img, src_pt1, src_pt2, (0, 255, 0), 3)
            cv2.line(cv_img, src_pt2, src_pt3, (0, 255, 0), 3)
            cv2.line(cv_img, src_pt3, src_pt4, (0, 255, 0), 3)

            height, width = cv_img.shape[:2]
            combine_filter = cv2.resize(combine_filter, (width, height))
            yellow_filter_roi = cv2.resize(yellow_filter_roi, (width, height))
            red_filter_roi = cv2.resize(red_filter_roi, (width, height))

            combine_filter = cv2.cvtColor(combine_filter, cv2.COLOR_GRAY2BGR)
            top_row = np.hstack((cv_img, combine_filter)) 
            bottom_row = np.hstack((yellow_filter_roi, red_filter_roi)) 
            bottom_row = cv2.cvtColor(bottom_row, cv2.COLOR_GRAY2BGR)
            merged_image = np.vstack((top_row, bottom_row)) 

            # [수정 6] imshow는 디버깅용. SSH 환경에서 느리면 주석 처리 하세요.
            cv2.imshow("Merged Image", merged_image)
            
            # [수정 7] waitKey 제거 (메인 루프에서 제어)
            # cv2.waitKey(1)

    #---------------------------MAIN------------------------------------#
    def ctrl(self):
        class_sub.lkas() 

        # start timer
        if self.start: 
            self.mission3_count = time()
            self.start = False

        # change lidar
        if self.flag4: 
            print("----------------mission4-----------------")
            ## chage lidar parameta
            self.OBSTACLE_PERCEPTION_BOUNDARY = 10
            self.scan_Ldgree = 50
            self.flag4 = False

        if self.flag6:
            print("----------------mission6-----------------")
            ## chage lidar parameta
            self.OBSTACLE_PERCEPTION_BOUNDARY = 15
            self.scan_Ldgree = 25
            self.v2x_flag = True
            self.flag6 = False 
            
        # lidar function
        if self.is_scan == True:
            first, first_dst, last, last_dst = self.LiDAR_scan()
            self.compare_space(first_dst, last_dst)
            self.move_direction(last, first)

        # final cmd_vel
        if self.obstacle_flag == True: 
            # print("====lidar")
            self.cmd_msg.linear.x = self.speed
            self.cmd_msg.angular.z = self.angle
        else :
            # print("====camera")
            self.cmd_msg.linear.x = self.camera_speed
            self.cmd_msg.angular.z = self.steer

        #---------------------------------------------V 2 X mission---------------------------------------------#
        current_time = time()

        ## modify 
        if self.v2x_flag: 
            if self.v2x == "D":
                self.camera_speed = 0 # stop the limo 
            else :
                self.camera_speed = 0.15
                self.v2x_flag = False 
                self.mission_ABC = True
                self.ABC_time = time()

        # #mission ABC
        if self.mission_ABC:
            if current_time - self.ABC_time < 2 and current_time - self.ABC_time > 0:
                if self.v2x == "A":
                    self.cmd_msg.angular.z = 0.55
                    print("turn to A")
            elif current_time - self.ABC_time < 4 and current_time - self.ABC_time > 2 :
                if self.v2x == "B":
                    print("turn to B")
                    self.cmd_msg.angular.z = 0.25 
            elif current_time - self.ABC_time < 7 and current_time - self.ABC_time > 6:
                if self.v2x == "C":
                    print("turn to C")
                    self.cmd_msg.angular.z = -0.15 

            elif self.ABC_time - current_time > 7 and self.ABC_time > 0:
                self.mission_ABC = False
        
        #---------------------------------------OBSTACLE AVOIDANCE--------------------------------------#  

        # A pass ==================================================================================
        if current_time - self.mission3_count > 23.5 and current_time - self.mission3_count < 25.5:
            if self.is_scan == False:
                print("----------------mission3-----------------")
            self.is_scan = True
            self.cmd_msg.angular.z = -0.1
            self.camera_speed = 0.15
            self.steer_weight = 1.7
        
        if current_time - self.mission3_count > 37.5 and current_time - self.mission3_count < 38.5:
            self.cmd_msg.angular.z = -0.25

        # B pass, C pass 주석 처리 됨 (필요시 해제)

        self.pub.publish(self.cmd_msg)
        self.rate.sleep()


if __name__ == "__main__":
    
    class_sub = Class_sub()

    # 초기 대기 화면 설정
    img = np.ones((300, 500), dtype=np.uint8) * 255 
    cv2.imshow("readytogo", img)
    
    print("========================================")
    print(" [S] Start Driving")
    print(" [K] Kill Process (Exit)")
    print(" [E] Emergency Stop (Reset Mission)")
    print("========================================")

    # 1. 시작 대기 루프 (Start Waiting Loop)
    start_driving = False
    while not rospy.is_shutdown():
        # 1ms 대기하며 키 입력 받기
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("LOG: Start Driving...")
            start_driving = True
            break
        elif key == ord('k'):
            print("LOG: Process Killed.")
            start_driving = False
            break

    # 2. 메인 주행 루프 (Main Control Loop)
    if start_driving:
        while not rospy.is_shutdown():
            class_sub.ctrl()
            
            # [수정 8] 키 입력을 변수에 저장 후 비교 (키 씹힘 방지)
            key = cv2.waitKey(1) & 0xFF 
            
            if key == ord('k'):
                print("LOG: Exiting by user request...")
                break
            
            elif key == ord('e'):
                print("LOG: Emergency Function Called!")
                class_sub.emergency()

        # 3. 종료 시 안전장치: 로봇 정지 명령 전송
        stop_msg = Twist()
        stop_msg.linear.x = 0
        stop_msg.angular.z = 0
        class_sub.pub.publish(stop_msg)
        print("LOG: Robot Stopped Safely.")

    # 윈도우 정리 및 노드 종료
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Finished")
