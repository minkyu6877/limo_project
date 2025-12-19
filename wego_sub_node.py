#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class UltimateGapRunner:
    def __init__(self):
        rospy.init_node("ultimate_gap_runner")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.camera_topic = "/usb_cam/image_raw"
        rospy.Subscriber(self.camera_topic, Image, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
        
        self.bridge = CvBridge()

        # ================= [기본] 주행 파라미터 =================
        self.drive_speed = 0.15
        self.steer_weight = 4.0 

        # ================= [NEW] 미션 6 (흰선) 제어 파라미터 =================
        self.base_gain = 1.0 / 200.0  
        self.corner_scale = 120.0     
        self.max_steer = 0.9          
        self.last_ang = 0.0           
        self.max_ang_step = 0.12      

        # ================= [NEW] 미션 3 (회피) 변수 =================
        self.scan_ranges = []
        self.front = 999.0
        
        self.state_start = 0.0
        self.escape_angle = 0.0
        self.left_escape_count = 0
        self.force_right_escape = 0
        
        # ★ 로봇 물리 정보 (미터 단위)
        self.robot_width = 0.18       # 로봇 폭 (0.13) + 여유 (0.05)
        self.obstacle_threshold = 0.6 # 이 거리보다 멀어야 '빈 공간'으로 인정

        # ================= 상태 관리 =================
        self.state = "LANE" 
        self.mission6_start_time = 0.0 

        # ================= HSV 설정 =================
        self.red_lower1 = np.array([0, 40, 50])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([165, 40, 50])
        self.red_upper2 = np.array([180, 255, 255])

        self.black_lower = np.array([102, 0, 60])
        self.black_upper = np.array([164, 86, 136])
        self.black2_lower = np.array([126, 25, 45])
        self.black2_upper = np.array([167, 89, 108])
        self.black3_lower = np.array([125, 29, 26])
        self.black3_upper = np.array([171, 100, 78])

        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([40, 255, 255])

        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 40, 255])

    # ============================================================
    # LIDAR Callback
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw # 전체 데이터 저장 (Gap Finding용)
        
        # 전방 20도 거리 측정 (충돌 감지용)
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.10 and not np.isnan(d)]
        
        if cleaned:
            self.front = np.median(cleaned) 
        else:
            self.front = 999.0

    # ============================================================
    # CAMERA Callback
    # ============================================================
    def camera_cb(self, msg):
        if self.state == "ESCAPE":
            self.escape_control()
            return
        if self.state == "BACK":
            self.back_control()
            return
        if self.state == "FINISH_6":
            self.run_blind_forward()
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e: return

        y, x, _ = cv_img.shape
        roi_h = int(y * 0.4)
        roi_img = cv_img[y - roi_h : y, 0 : x]
        hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # 마스크 생성
        mask_yellow = cv2.inRange(hsv_img, self.yellow_lower, self.yellow_upper)
        mask_white = cv2.inRange(hsv_img, self.white_lower, self.white_upper)
        
        mask_r1 = cv2.inRange(hsv_img, self.red_lower1, self.red_upper1)
        mask_r2 = cv2.inRange(hsv_img, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        
        mask_b1 = cv2.inRange(hsv_img, self.black_lower, self.black_upper)
        mask_b2 = cv2.inRange(hsv_img, self.black2_lower, self.black2_upper)
        mask_b3 = cv2.inRange(hsv_img, self.black3_lower, self.black3_upper)
        mask_bk = cv2.bitwise_or(mask_b1, mask_b2)
        mask_black = cv2.bitwise_or(mask_bk, mask_b3)
        
        kernel = np.ones((3, 3), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)

        # --- MISSION 6 ---
        if self.state == "MISSION_6":
            contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours_white if cv2.contourArea(c) > 200]

            if len(valid_contours) == 0:
                rospy.loginfo("White Line Ends -> Finish 30cm")
                self.state = "FINISH_6"
                self.mission6_start_time = rospy.Time.now().to_sec()
                self.publish_cmd(0.0)
                return

            cx_sum = 0
            weight_sum = 0
            for c in valid_contours:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    c_x = int(M['m10'] / M['m00'])
                    area = M['m00']
                    cx_sum += c_x * area
                    weight_sum += area
            
            if weight_sum == 0: return

            target_cx = int(cx_sum / weight_sum) 
            center_x = x // 2                    
            error = target_cx - center_x         

            gain = self.base_gain / (1.0 + abs(error) / self.corner_scale)
            target_w = gain * error

            dw = target_w - self.last_ang
            dw = max(min(dw, self.max_ang_step), -self.max_ang_step)

            final_ang = self.last_ang + dw
            final_ang = max(min(final_ang, self.max_steer), -self.max_steer)
            self.last_ang = final_ang 

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = final_ang
            self.pub.publish(twist)
            return

        # --- LANE MODE ---
        
        # 1. 노란색 감지
        if cv2.countNonZero(mask_yellow) > 1000:
            rospy.logwarn("YELLOW DETECTED -> MISSION 6")
            self.state = "MISSION_6"
            self.last_ang = 0.0
            self.publish_cmd(0.0)
            return

        # 2. 빨간색 우선
        if cv2.countNonZero(mask_red) > 500:
            left_red = mask_red[:, :x//2]
            right_red = mask_red[:, x//2:]
            balance_error = cv2.countNonZero(right_red) - cv2.countNonZero(left_red)
            steer = (balance_error / (x * roi_h * 0.1)) * 2.0
            self.publish_cmd(steer)
            return 

        # 3. 전방 장애물 감지 -> BACK MODE 진입
        if self.front < 0.40:
            rospy.logwarn(f"OBSTACLE ({self.front:.2f}m) -> BACK MODE")
            self.state = "BACK"
            self.state_start = rospy.Time.now().to_sec()
            return

        # 4. 검은색 차선
        if cv2.countNonZero(mask_black) > 500: 
            warp_img = self.get_bev(mask_black, x, roi_h)
            steer = self.calculate_steer(warp_img)
            self.publish_cmd(steer)
        else:
            self.publish_cmd(0.0)

    # ============================================================
    # [수정됨] 회피 로직: BACK -> GAP FINDING (Center) -> ESCAPE
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        
        # 1.4초간 후진
        if now - self.state_start < 1.4:
            twist.linear.x = -0.24
            twist.angular.z = 0.0
            self.pub.publish(twist)
        else:
            # 후진 완료 후, "가장 좋은 빈 공간의 중앙"을 찾음
            angle = self.find_best_gap_center()
            
            # 갇힘 방지 (계속 한쪽으로만 도는 경우 대비)
            angle = self.apply_escape_direction_logic(angle)
            
            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now
            rospy.loginfo(f"Back Done -> Gap Center Angle: {math.degrees(angle):.2f} deg")

    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        
        # 1.0초간 계산된 각도로 회전하며 전진
        if now - self.state_start < 1.0:
            twist.linear.x = 0.19
            # 회전 각도에 가중치를 두어 확실하게 틀도록 함
            twist.angular.z = self.escape_angle * 1.5
            # 최대 회전 속도 제한
            twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
            self.pub.publish(twist)
        else:
            self.state = "LANE"
            self.last_ang = 0.0

    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.9 # 강제 우회전

        if angle < 0: # 왼쪽으로 피하려 할 때
            self.left_escape_count += 1
            if self.left_escape_count >= 3: # 3번 연속 왼쪽이면
                self.force_right_escape = 2 # 다음 2번은 오른쪽 강제
                self.left_escape_count = 0
        else:
            self.left_escape_count = 0

        return angle

    # ============================================================
    # [핵심] 빈 공간의 중앙(Center)을 찾는 알고리즘 
    # ============================================================
    def find_best_gap_center(self):
        if len(self.scan_ranges) == 0:
            return 0.0
        
        raw = np.array(self.scan_ranges)
        # 전방 120도 (-60도 ~ +60도) 스캔
        # 배열 순서: [우측 데이터 ... 정면 ... 좌측 데이터] 일 수 있으니 인덱스 확인 필요
        # 보통 Limo는 인덱스 0이 뒤쪽일 수 있으므로 슬라이싱 주의.
        # 여기서는 기존 로직대로 raw[-60:] + raw[:60] (뒤쪽이 0도 기준일 때 전방) 사용
        ranges = np.concatenate([raw[-60:], raw[:60]])
        
        # 1. 데이터 전처리 (노이즈 제거 및 거리 제한)
        # self.obstacle_threshold(0.6m)보다 멀면 '빈 공간(Open)', 가까우면 '벽(Wall)'
        # 3.0m 이상은 3.0으로 통일 (무한대 방지)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)
        ranges = np.where(np.isinf(ranges), 3.0, ranges)
        
        # 2. Gap(연속된 빈 공간) 찾기
        gaps = []
        current_gap_start = -1
        
        for i, dist in enumerate(ranges):
            if dist > self.obstacle_threshold:
                # 빈 공간 시작
                if current_gap_start == -1:
                    current_gap_start = i
            else:
                # 빈 공간 끝남 -> Gap 저장
                if current_gap_start != -1:
                    gaps.append((current_gap_start, i - 1))
                    current_gap_start = -1
        
        # 마지막 Gap 처리
        if current_gap_start != -1:
            gaps.append((current_gap_start, len(ranges) - 1))
            
        # 3. 유효한 Gap 필터링 (로봇이 지나갈 수 있는지?)
        valid_gaps = []
        for start, end in gaps:
            gap_len = end - start
            if gap_len < 3: continue # 노이즈 제거
            
            # Gap의 평균 거리 계산
            gap_ranges = ranges[start:end+1]
            avg_dist = np.mean(gap_ranges)
            
            # 실제 너비(Arc Length) 계산: 거리 * 각도차이(라디안)
            # 인덱스 1개당 1도(약 0.0174라디안)라고 가정
            angle_span_rad = gap_len * (math.pi / 180.0)
            real_width = avg_dist * angle_span_rad
            
            # 로봇 폭보다 넓은지 확인
            if real_width > self.robot_width:
                valid_gaps.append({'start': start, 'end': end, 'width': real_width})

        # 4. 최적의 Gap 선택 (가장 넓은 곳)
        if not valid_gaps:
            return 0.0 # 갈 곳이 없으면 정지 혹은 직진

        best_gap = max(valid_gaps, key=lambda x: x['width'])
        
        # 5. Gap의 중앙 인덱스 계산
        center_idx = (best_gap['start'] + best_gap['end']) / 2.0
        
        # 6. 인덱스를 각도로 변환 (-60도 ~ +60도 범위)
        # idx 0 = -60도, idx 60 = 0도, idx 120 = +60도
        target_angle_rad = (center_idx - 60) * (math.pi / 180.0)
        
        return target_angle_rad

    # ============================================================
    # MISSION 6 종료
    # ============================================================
    def run_blind_forward(self):
        now = rospy.Time.now().to_sec()
        elapsed = now - self.mission6_start_time
        if elapsed < 2.0:
            twist = Twist()
            twist.linear.x = 0.15 
            twist.angular.z = 0.0 
            self.pub.publish(twist)
        else:
            rospy.loginfo("Mission 6 Done -> Lane Mode")
            self.state = "LANE"
            self.last_ang = 0.0

    # --- 유틸리티 함수 ---
    def publish_cmd(self, steer):
        twist = Twist()
        twist.linear.x = self.drive_speed
        twist.angular.z = np.clip(steer, -1.0, 1.0)
        self.pub.publish(twist)

    def get_bev(self, mask, width, height):
        src_pts = np.float32([(10, height), (width//2 - 60, 0), (width//2 + 60, 0), (width - 10, height)])
        dst_w, dst_h = 400, 400
        dst_pts = np.float32([(100, dst_h), (100, 0), (dst_w - 100, 0), (dst_w - 100, dst_h)])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(mask, M, (dst_w, dst_h))

    def calculate_steer(self, warp_img):
        M = cv2.moments(warp_img)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            error = 200 - cx 
            return (error / 200.0) * self.steer_weight
        return 0.0

if __name__ == "__main__":
    try:
        UltimateGapRunner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
