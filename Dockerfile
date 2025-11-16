# 1. 베이스 이미지 설정 (LIMO의 ROS 버전에 맞춤, 보통 Melodic)
FROM ros:melodic-ros-base

# 2. 필수 패키지 설치 (OpenCV, ROS Bridge 등)
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-melodic-cv-bridge \
    ros-melodic-sensor-msgs \
    ros-melodic-geometry-msgs \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3. Python 라이브러리 설치
RUN pip3 install opencv-python numpy

# 4. 작업 디렉토리 생성
WORKDIR /app

# 5. 내 코드를 이미지 안으로 복사
COPY wego_sub_node.py /app/wego_sub_node.py

# 6. 실행 권한 부여
RUN chmod +x /app/wego_sub_node.py

# 7. 환경 변수 설정 (ROS 마스터와 통신하기 위함)
# 실제 실행 시 docker run 명령어에서 덮어씌워질 수 있음
CMD ["/bin/bash", "-c", "source /opt/ros/melodic/setup.bash && python3 wego_sub_node.py"]
