#!/bin/bash
set -e

# 支持的输入格式
MMDET="ON"
MMCLS="ON"

# 支持的输出格式，ONNX默认支持
MNN="ON"
CAFFE="ON"
NNIE="ON"
TENSORRT="ON"
TENGINE="ON"

# auto-det git地址
MMDET_GIT="https://gitee.com/mirrors/mmdetection.git"
# auto-cls git地址
MMCLS_GIT="https://github.com/open-mmlab/mmclassification.git"
# Caffe git地址
CAFFE_GIT="https://gitee.com/ml-inory/Caffe.git"
# MNN git地址
MNN_GIT="https://github.com/alibaba/MNN.git"
# NNIE git地址
NNIE_GIT="https://gitee.com/ml-inory/nnie_mapper.git"
# OpenCV git地址
OPENCV_GIT="https://gitee.com/mirrors/opencv.git"
# Tengine git地址
TENGINE_GIT="https://github.com/OAID/Tengine-Convert-Tools.git"

# auto-det目录名称
MMDET_NAME="mmdet"
# auto-cls目录名称
MMCLS_NAME="mmcls"
# caffe目录名称
CAFFE_NAME="caffe"
# MNN目录名称
MNN_NAME="mnn"
# NNIE目录名称
NNIE_NAME="nnie_mapper"
# OpenCV目录名称
OPENCV_NAME="opencv"
# TensorRT目录名称
TENSORRT_NAME="tensorrt"
# Tengine目录名称
TENGINE_NAME="tengine"

# auto-det git分支
MMDET_GIT_BRANCH="master"
# auto-cls git分支
MMCLS_GIT_BRANCH="master"
# caffe git分支
CAFFE_GIT_BRANCH="master"
# MNN git分支
MNN_GIT_BRANCH="master"
# NNIE git分支
NNIE_GIT_BRANCH="master"
# OpenCV git分支
OPENCV_GIT_BRANCH="3.4"
# Tengine git分支
TENGINE_GIT_BRANCH="master"

# git仓库是否自动更新
GIT_AUTO_PULL="ON"

# git账号
GIT_USER=""
GIT_PASS=""

# 第三方库安装目录
EXT_DIR=external
# 第三方库安装路径
EXT_INSTALL_PREFIX=/usr/local


# 目录转换为绝对路径
EXT_DIR=$(readlink -f $EXT_DIR)
# 当前目录
CUR_DIR=$(readlink -f .)

# 安装pycocotools
function install_pycocotools() {
    pip3 install cython
    # 仓库已存在
    if [ ! -d pycocotools ]; then
        git clone https://github.com/open-mmlab/cocoapi.git pycocotools --depth=1
    fi
    cd pycocotools/pycocotools
    python3 setup.py install
    cd ../..
}

# 添加环境变量
function export_var() {
    if [ `grep -c "$1" /etc/profile` -eq '0' ]; then
        echo $1 >> /etc/profile
    fi
}

function export_ld() {
    if [ `grep -c "$1" /etc/profile` -eq '0' ]; then
        echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$1" >> /etc/profile
    fi
}

function export_path() {
    if [ `grep -c "$1" /etc/profile` -eq '0' ]; then
        echo "export PATH=$PATH:$1" >> /etc/profile
    fi
}

# 安装mmdetection
function install_mmdet() {
    if [ "ON" == $MMDET ]; then
        echo "================ INSTALL MMDET ================"
        # 仓库已存在
        if [ -d $MMDET_NAME ]; then
            echo "$MMDET_NAME exists."
            cd $MMDET_NAME
            # 检查git分支是否一致
            if [ `git symbolic-ref --short -q HEAD` != $MMDET_GIT_BRANCH ]; then
                echo "Current branch mismatch, change to $MMDET_GIT_BRANCH"
                git checkout $MMDET_GIT_BRANCH
            fi
            # 更新仓库
            if [ "ON" == $GIT_AUTO_PULL ]; then
                git pull
            fi
        else
            echo "$MMDET_NAME not exist, clone from git."
            git clone -b $MMDET_GIT_BRANCH $MMDET_GIT $MMDET_NAME
            cd $MMDET_NAME
        fi

        pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
        pip3 install pytest-runner
        pip3 install "mmcv-full>=1.1.1, <=1.2" --trusted-host mirrors.cloud.aliyuncs.com
        install_pycocotools
        pip3 install -r requirements.txt
        python3 setup.py develop
    fi 
    cd $EXT_DIR
}

# 安装mmcls
function install_mmcls() {
    if [ "ON" == $MMCLS ]; then
        echo "================ INSTALL MMCLS ================"
        # 仓库已存在
        if [ -d $MMCLS_NAME ]; then
            cd $MMCLS_NAME
            # 检查git分支是否一致
            if [ `git symbolic-ref --short -q HEAD` != $MMCLS_GIT_BRANCH ]; then
                git checkout $MMCLS_GIT_BRANCH
            fi
            # 更新仓库
            if [ "ON" == $GIT_AUTO_PULL ]; then
                git pull
            fi
        else
            git clone -b $MMCLS_GIT_BRANCH $MMCLS_GIT $MMCLS_NAME
            cd $MMCLS_NAME
        fi

        pip3 install -r requirements.txt
        pip3 install filelock
        python3 setup.py develop
    fi 
    cd $EXT_DIR
}

# 安装ONNX
function install_onnx() {
    echo "================ INSTALL ONNX ================"
    pip3 install onnx onnx-simplifier onnxruntime --use-feature=2020-resolver
    cd $EXT_DIR
}

# 安装OpenCV
function install_opencv() {
    echo "================ INSTALL OPENCV ================"
    # 仓库已存在
    if [ -d $OPENCV_NAME ]; then
        cd $OPENCV_NAME
        # 检查git分支是否一致
        if [ `git symbolic-ref --short -q HEAD` != $OPENCV_GIT_BRANCH ]; then
            git checkout $OPENCV_GIT_BRANCH
        fi
    else
        git clone -b $OPENCV_GIT_BRANCH $OPENCV_GIT $OPENCV_NAME
        cd $OPENCV_NAME
    fi

    apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx
    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=$EXT_INSTALL_PREFIX -DWITH_IPP=OFF -DBUILD_opencv_python=OFF -DBUILD_opencv_python3=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF ..
    make -j`nproc` && make install

    cd $EXT_DIR
}

# 安装Caffe
function install_caffe() {
    if [ "ON" == $CAFFE ]; then
        echo "================ INSTALL CAFFE ================"
        # 仓库已存在
        if [ -d $CAFFE_NAME ]; then
            cd $CAFFE_NAME
            # 检查git分支是否一致
            if [ `git symbolic-ref --short -q HEAD` != $CAFFE_GIT_BRANCH ]; then
                git checkout $CAFFE_GIT_BRANCH
            fi
            # 更新仓库
            if [ "ON" == $GIT_AUTO_PULL ]; then
                git pull
            fi
        else
            git clone -b $CAFFE_GIT_BRANCH $CAFFE_GIT $CAFFE_NAME
            cd $CAFFE_NAME
        fi

        apt-get install -y cmake build-essential libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
        apt-get install -y python3-dev python3-numpy python3-pip
        pip3 install opencv-python numpy cython
        apt-get install -y --no-install-recommends libboost-all-dev
        apt-get install -y libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
        mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$EXT_INSTALL_PREFIX -DCPU_ONLY=ON -Dpython_version=3 .. && make -j`nproc` && make pycaffe && make install
        pip3 install -r $EXT_INSTALL_PREFIX/python/requirements.txt
        export PYTHONPATH=$EXT_INSTALL_PREFIX/python
        export_var "export PYTHONPATH=$EXT_INSTALL_PREFIX/python"
    fi 
    cd $EXT_DIR
}

# 安装MNN
function install_mnn() {
    if [ "ON" == $MNN ]; then
        echo "================ INSTALL MNN ================"
        # 仓库已存在
        if [ -d $MNN_NAME ]; then
            cd $MNN_NAME
            # 检查git分支是否一致
            if [ `git symbolic-ref --short -q HEAD` != $MNN_GIT_BRANCH ]; then
                git checkout $MNN_GIT_BRANCH
            fi
            # 更新仓库
            if [ "ON" == $GIT_AUTO_PULL ]; then
                git pull
            fi
        else
            git clone -b $MNN_GIT_BRANCH $MNN_GIT $MNN_NAME
            cd $MNN_NAME
        fi

        if [ ! -f ./schema/current/MNN_generated.h ]; then
            cd schema && source generate.sh && cd ..
        fi
        mkdir -p build && cd build
        cmake -DCMAKE_INSTALL_PREFIX=$EXT_INSTALL_PREFIX -DMNN_BUILD_CONVERTER=ON ..
        make -j`nproc` && make install
    fi 
    cd $EXT_DIR
}

# 安装NNIE
function install_nnie() {
    if [ "ON" == $NNIE ]; then
        echo "================ INSTALL NNIE ================"
        apt-get install gcc-4.8 g++-4.8 -y
        apt-get install -y libjpeg-dev libpng-dev libtiff-dev
        # 仓库已存在
        if [ -d $NNIE_NAME ]; then
            cd $NNIE_NAME
            # 检查git分支是否一致
            if [ `git symbolic-ref --short -q HEAD` != $NNIE_GIT_BRANCH ]; then
                git checkout $NNIE_GIT_BRANCH
            fi
            # 更新仓库
            if [ "ON" == $GIT_AUTO_PULL ]; then
                git pull
            fi
        else
            git clone -b $NNIE_GIT_BRANCH $NNIE_GIT $NNIE_NAME
            cd $NNIE_NAME
        fi
        chmod 777 -R ./
        chmod +x nnie_mapper_12
        export NNIE_MAPPER=$(readlink -f ./)
        export_var "export NNIE_MAPPER=$(readlink -f ./)"
    fi 
    cd $EXT_DIR
}

# 安装TensorRT
function install_tensorrt() {
    echo "================ INSTALL TENSORRT ================"
    if [ -d /usr/local/cuda ]; then
        if [ "ON" == $TENSORRT ]; then
            if [ ! -d $TENSORRT_NAME ]; then
                TAR_NAME=$TENSORRT_NAME.tar.gz
                wget -t 0 -c -O $TAR_NAME "https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/tars/TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz"
                mkdir -p $TENSORRT_NAME && tar -xzvf $TAR_NAME -C $TENSORRT_NAME --strip-components 1
                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_NAME/lib
                export_ld $TENSORRT_NAME/lib
                cd $TENSORRT_NAME/python
                pip3 install tensorrt-*-cp3x-none-linux_x86_64.whl
                cd ../graphsurgeon
                pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl
                cd ../onnx_graphsurgeon
                pip3 install onnx_graphsurgeon-0.2.0-py2.py3-none-any.whl
                apt-get install python3-libnvinfer-dev
            else
                echo "Already installed."
            fi
        fi 
    else
        echo "CUDA not available, ignore TENSORRT"
        TENSORRT="OFF"
    fi
    cd $EXT_DIR
}

# 安装Tengine
function install_tengine() {
    if [ "ON" == $TENGINE ]; then
        echo "================ INSTALL Tengine ================"
        # 仓库已存在
        if [ -d $TENGINE_NAME ]; then
            cd $TENGINE_NAME
            # 检查git分支是否一致
            if [ `git symbolic-ref --short -q HEAD` != $TENGINE_GIT_BRANCH ]; then
                git checkout $TENGINE_GIT_BRANCH
            fi
            # 更新仓库
            if [ "ON" == $GIT_AUTO_PULL ]; then
                git pull
            fi
        else
            git clone -b $TENGINE_GIT_BRANCH $TENGINE_GIT $TENGINE_NAME
            cd $TENGINE_NAME
        fi

        apt install -y libprotobuf-dev protobuf-compiler

        mkdir -p build && cd build
        cmake -DCMAKE_INSTALL_PREFIX=$EXT_INSTALL_PREFIX ..
        make -j4 && make install
    fi 
    cd $EXT_DIR
}
# 安装第三方库
function install_dependency() {
    echo "================ install dependency to $EXT_DIR ================"
    mkdir -p $EXT_DIR
    cd $EXT_DIR

    apt-get install -y python3-pip cmake build-essential
    python3 -m pip install --upgrade pip
    pip3 config set global.default-timeout 1000
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    git config --global http.sslverify false
    
    install_opencv
    install_mmdet
    install_mmcls
    install_onnx
    install_caffe
    install_mnn
    install_nnie
    install_tensorrt

    echo "================ install dependency finish ================"
}

# 安装工具
function install_tools() {
    echo "================ install tools ================"
    cd $CUR_DIR

    if [ "ON" == $MNN ]; then
        export PATH=$PATH:$(readlink -f $EXT_DIR/$MNN_NAME/build)
        export_path $(readlink -f $EXT_DIR/$MNN_NAME/build)
        export SETUP_MNN=ON
    fi

    if [ "ON" == $TENGINE ]; then
        export PATH=$PATH:$(readlink -f $EXT_DIR/$TENGINE_NAME/build/install/bin)
        export_path $(readlink -f $EXT_DIR/$TENGINE_NAME/build/install/bin)
        export SETUP_TENGINE=ON
    fi

    sudo python3 setup.py install
    source /etc/profile
}

install_dependency
install_tools