# road_crossing_jetson_nano

## Install General Packages
  sudo apt-get install python3-opencv
  sudo apt-get install python3-pandas
  sudo apt-get install python3-numpy
  sudo apt-get install python3-scipy
  sudo apt-get install python3-matplotlib

## Install Scipy
python3 -m pip install --upgrade pip
pip install scipy

## Install Detectron2 
https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048
### Install Packages
#### PyTorch 1.8.0
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#### Torchvision 0.9.0
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch <version> https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
python3 setup.py install --user
cd ../

#### Other Dependencies
pip3 install cython 
pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install pyyaml --upgrade

## For Opencv imshow
sudo apt-get install libcanberra-gtk-module
