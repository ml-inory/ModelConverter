# 模型转换全家桶

包括：  
- mmdet2onnx
- mmcls2onnx
- onnx2caffe
- onnx2mnn
- onnx2tflite
- onnx2trt
- caffe2nnie

## 安装

`sudo bash build.sh`

出现输入提示后输入共达地Gitlab的账号密码即可

## 使用

可通过命令行或python3 -m的方式使用，如：

`mmdet2onnx` 或 `python3 -m mmdet2onnx`

## 测试

tests目录下运行`python3 test.py`