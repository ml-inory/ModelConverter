# -*- coding: utf-8 -*-
'''
*******************************************************************************
* File: onnx2trt.py
* Author: rzyang
* Date: 2020/08/03
* Description: Convert ONNX PyTorch model to TensorRT and check result.
*******************************************************************************
'''

import os, sys

if not os.path.exists('/usr/local/cuda'):
    print('CUDA is not available, exit!')
    sys.exit(-1)

import numpy as np

# ONNX
import onnx
from onnxsim import simplify
import onnxruntime

# TensorRT
import tensorrt as trt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ONNX model to TensorRT model Converter')
    parser.add_argument('--onnx', help='onnx model file path, *.onnx', required=True)
    parser.add_argument('--trt', help='tensorrt model file path, *.trt', required=True)
    parser.add_argument('--fp16', help='open tensorrt fp16 mode', action='store_true')
    parser.add_argument('--debug', help='print out args', action='store_true')
    args = parser.parse_args()
    return args

def debug(args):
    print('=' * 40 + ' debug ' + '=' * 40)
    print('onnx: {}\ntensorrt: {}\nFP16: {}\n\n'.format(args.onnx, args.trt, args.fp16))

def convert(onnx_file, trt_file, fp16):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        # FP16
        builder.fp16_mode = fp16
        builder.max_workspace_size = 1 << 25

        print('Loading ONNX file from path {}...'.format(onnx_file))
        with open(onnx_file, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(error, parser.get_error(error))
                raise Exception('onnx model parsing error while converting to tensorrt')
            print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_file))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            # save plan file
            with open(trt_file, "wb") as f:
                f.write(engine.serialize())

    print('Convert {} to {} success'.format(onnx_file, trt_file))

def main():
    args = parse_args()
    if args.debug:
        debug(args)
    convert(args.onnx, args.trt, args.fp16)

if __name__ == '__main__':
    main()
    