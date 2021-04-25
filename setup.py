#!/usr/bin/env python3
import os, sys
from setuptools import setup, find_packages

if __name__ == '__main__':
    environ = os.environ
    modules = ('mmdet2onnx', 'mmcls2onnx', 'onnx2trt', 'onnx2caffe', 'caffe2nnie', 'onnx2mnn', 'onnx2tengine')

    install_modules = []
    console_scripts = []
    for module in modules:
        install_modules.append(module)
        console_scripts.append(f'{module} = {module}.__main__:main')
        print(f'Install {module} = {module}:main')
    install_modules.extend(find_packages())

    setup(
        name        = 'ModelConverter',
        version     = '1.0',
        author      = 'rzyang',
        author_email= 'madscientist_yang@foxmail.com',
        url         = 'https://github.com/ml-inory/ModelConverter',
        description = 'All model converters',
        packages    = install_modules,
        package_data = {'': ['onnx2caffe_src']},
        entry_points={
            'console_scripts': console_scripts
      },
    )
