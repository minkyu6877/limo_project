# 1. 베이스 이미지: LIMO 표준인 ROS Noetic 사용
FROM ros:noetic-ros-base

# 2. 필수 패키지 설치 (영상 처리 및 시스템 도구)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-vision-opencv \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 파이썬 라이브러리 설치
RUN pip3 install numpy PyYAML rospkg

# 4. 작업 디렉토리 설정 (ROS 표준 경로)
WORKDIR /root/catkin_ws/src

# 5. 접속 시 자동 경로 이동 설정 (편의용)
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "cd /root/catkin_ws/src" >> ~/.bashrc
