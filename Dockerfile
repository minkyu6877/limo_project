# 1. 베이스 이미지 설정
FROM ros:melodic-ros-base

# 2. 필수 패키지 설치
# [핵심 수정] python3-opencv를 여기서 설치합니다. (빌드 에러 해결)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-melodic-cv-bridge \
    ros-melodic-sensor-msgs \
    ros-melodic-geometry-msgs \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3. Python 라이브러리 설치
# opencv-python은 위에서 설치했으므로 제거했습니다. numpy만 pip로 설치합니다.
RUN pip3 install numpy PyYAML rospkg

# 4. 작업 디렉토리 생성
WORKDIR /app

# 5. 내 코드를 이미지 안으로 복사
COPY wego_sub_node.py /app/wego_sub_node.py

# 6. 실행 권한 부여
RUN chmod +x /app/wego_sub_node.py

# 7. 실행 명령 설정
CMD ["/bin/bash", "-c", "source /opt/ros/melodic/setup.bash && python3 wego_sub_node.py"]
