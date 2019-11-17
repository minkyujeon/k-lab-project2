

## Openpose_CMU



### 환경 설정

https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md

여기에서 Installation 부분만 보면 됨. (거의 볼 필요없음)



https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/prerequisites.md (여기에서 환경세팅하면 됩니다.)

sudo apt-get install cmake-qt-gui

#sudo apt-get install cmake-qt-gui

sudo apt purge cmake-qt-gui

sudo apt-get install qtbase5

#cmake 최신버전 다운.  cmake-X.X.X.tar.gz

./configure --qt

./bootstrap && make -j`nproc` && sudo make install -j`nproc` 

sudo apt-get install libviennacl-dev

sudo bash ./scripts/ubuntu/install_deps.sh

마지막으로 Eigen prerequisite - 아직 설치 안한상태.

환경만 맞춰주면 사용할 수 있기에 위의 링크를 통해서 환경설정을 해줘야 한다.

Cuda 10.0, cudnn 7.x로 설치.(정확한 버전은 서버열고 확인해볼 것.) 

cmake 부분이 가장 어려웠는데 맨 처음에는 cmake-gui로 하지 않고 cmake를 따로 만들어줌. 





https://evols-atirev.tistory.com/27

gcp gui는 여기 링크에서 하라는대로 천천히 하면 됨.

그리고 xrdp를 설치하고 나서 window에서는 원격 데스크톱 열고 서버 ip를 통해서 연결하면 됨.

그리고 맨 처음에 비밀번호를 다시 설정하라고 하는데 다시 세팅을 해야 gui ubuntu에서 비밀번호 사용할 수 있음. 나는 그냥 서버 비밀번호 그대로 다시 설정.





### 결과

1. demo version

- test 결과만 gui로 보는 것.

  - ```
    ./build/examples/openpose/openpose.bin --video examples/media/video.avi
    ```

- gui없이 json만 저장. 이게 맞음.

  - ./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_json output/ --display 0 --render_pose 0

- yum 형태로 저장

  - ./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_keypoint output/ --display 0 --render_pose 0



output 형태

- pose_keypoints_2d:  (x1,y1,c1)    x,y좌표 c: confidence

- Result for BODY_25

  - ```
         {0,  "Nose"},
         {1,  "Neck"},
         {2,  "RShoulder"},
         {3,  "RElbow"},
         {4,  "RWrist"},
         {5,  "LShoulder"},
         {6,  "LElbow"},
         {7,  "LWrist"},
         {8,  "MidHip"},
         {9,  "RHip"},
         {10, "RKnee"},
         {11, "RAnkle"},
         {12, "LHip"},
         {13, "LKnee"},
         {14, "LAnkle"},
         {15, "REye"},
         {16, "LEye"},
         {17, "REar"},
         {18, "LEar"},
         {19, "LBigToe"},
         {20, "LSmallToe"},
         {21, "LHeel"},
         {22, "RBigToe"},
         {23, "RSmallToe"},
         {24, "RHeel"},
         {25, "Background"}
    ```

- output의 정보가 어떤 사람꺼냐는는 frame에서 왼쪽 사람부터.