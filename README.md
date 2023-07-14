# Object Tracking OpenCV

Small OpenCV project for object detection and tracking.

## Environment

1. OS: Ubuntu 20.04
2. GPU: NVIDIA GeForce RTX 3050 Ti Laptop
3. CPU: AMD Ryzen 7 5800H with Radeon Graphics
4. CUDA: CUDA 11.4
5. CUDNN: CUDNN 8.9.2.26-1+cuda11.8
6. OpenCV: OpenCV 4.5.4
7. pyTorch: torch 1.11.0+cu113
8. torchvision: torchvision 0.12.0+cu113

## Setup

### CUDA

CUDA is optional but recommended to speed-up inference and training speeds with OpenCV and pyTorch

[CUDA Install Instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)

[CUDA Download](https://developer.nvidia.com/cuda-toolkit-archive)

### CUDNN

CUDNN is optional but is required for OpenCV DNN acceleration

[CUDNN Install Instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

[CUDNN Download](https://developer.nvidia.com/rdp/cudnn-download)

### OpenCV

OpenCV and OpenCV contrib is required and has to be build from source if using CUDA and CUDNN acceleration.


1. Enter working directory
   ```sh
   cd /path/to/tracking/ws
   ```

2. Run setup.sh script
   ```sh
   sudo chmod +x setup.sh
   ./setup.sh
   ```

3. Install OpenCV and OpenCV contrib without CUDA and CUDNN
   ```sh
   sudo apt-get update
   sudo apt-get install libopencv-dev=4.5.4 libopencv-python=4.5.4
   ```

   OR

   Install OpenCV and OpenCV contrib with CUDA and CUDNN
   Please refer to documentation from [OpenCV](https://docs.opencv.org/4.5.4/d7/d9f/tutorial_linux_install.html) for install

   Use the below CMake script:
   Please do change what ever applicable i.e. 
   - `-DCMAKE_INSTALL_PREFIX`
   - `-DCUDA_TOOLKIT_ROOT_DIR`
   - `-DOPENCV_EXTRA_MODULES_PATH`

   ```sh
   cmake -DWITH_OPENGL=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_opencv_cudacodec=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/opencv-4.5.4-linux -DWITH_TBB=ON -DBUILD_EXAMPLES=OFF -DBUILD_opencv_world=OFF -DBUILD_opencv_gapi=ON -DBUILD_opencv_wechat_qrcode=OFF -DWITH_QT=ON -DWITH_OPENGL=ON -DWITH_GTK=ON -DWITH_GTK3=ON -DWITH_GTK_2_X=OFF -DWITH_VTK=OFF -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_CUBLAS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4 -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
   ```

   Open a terminal and export library path as per `-DCMAKE_INSTALL_PREFIX`
   ```sh
   echo 'export LD_LIBRARY_PATH=~/opencv-4.5.4-linux/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   echo 'export PYTHONPATH=~/opencv-4.5.4-linux/local/lib/python3.8/dist-packages/:$PYTHONPATH' >> ~/.bashrc
   source ~/.bashrc
   ```
   
### virtualenv
Setup python virtual environment to install pyTorch and YOLO python packages

```sh
sudo apt-get install virtualenv
cd ~/
virtualenv env_yolo --system-site-packages
echo 'export LD_LIBRARY_PATH=~/opencv-4.5.4-linux/local/lib:$LD_LIBRARY_PATH' >> ~/env_yolo/bin/activate
echo 'export PYTHONPATH=~/opencv-4.5.4-linux/local/lib/python3.8/dist-packages/:$PYTHONPATH' >> ~/env_yolo/bin/activate
source ~/env_yolo/bin/activate
```

### pyTorch

Install pyTorch and its compenent compatible with your CUDA version.
```sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### YOLOv5

Install YOLOv5 using instructions from their [documentation](https://docs.ultralytics.com/yolov5/quickstart_tutorial/)

For export .pt to .onnx

Warning: using --opset 11 for exporting to ONNX
```sh
python3 export.py --weights best.pt --include onnx --device 0 --opset 11
```
For training

Warning: using --batch-size 12 or lower for training
```sh
python train.py --img 640 --epochs 300 --data merge_class_random_split.yaml --weights yolov5s.pt --batch-size 64 --device 0 --optimizer AdamW --patience 50 --save-period 50
```

For detect
```sh
python detect.py --weights ~/yolov5/run/train/5_epoch_all/weights/best.pt --source ~/tracking_ws/videos/video1.avi --view-img
```

## Usage

To be updated, change CMake to include directory
