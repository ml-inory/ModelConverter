# -*- coding: utf-8 -*-
'''
*******************************************************************************
* File: mmdet2onnx.py
* Author: rzyang
* Date: 2020/08/03
* Description: Convert mmdetection PyTorch model to ONNX and check result.
*******************************************************************************
'''
import os
import numpy as np

# ONNX
import onnx
from onnxsim import simplify
import onnxruntime
import copy

# mmdetection
import torch
from mmdet.apis import init_detector

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='MMDetection PyTorch model to ONNX model Converter')
    parser.add_argument('--config', help='mmdetection config file path', required=True)
    parser.add_argument('--checkpoint', help='mmdetection .pth file path', required=False, default=None)
    parser.add_argument('--onnx', help='onnx model file path', required=True)
    parser.add_argument('--input_size', help='input size of model, format: height width channel', type=int, required=False, nargs='*')
    parser.add_argument('--device', help='device: cpu or cuda:0', required=False, default='cpu')
    parser.add_argument('--debug', help='print out args', action='store_true')
    args = parser.parse_args()
    return args


def debug(args):
    print('=' * 40 + ' debug ' + '=' * 40)
    print('config: {}\ncheckpoint: {}\nonnx: {}\ninput_size: {}\ndevice: {}\n\n'.format(args.config, args.checkpoint, args.onnx, args.input_size, args.device))

def rename_share_weights(onnx_file):
    model = onnx.load(onnx_file)
    
    tensor_to_del = []
    for t in model.graph.initializer:
        share = 0
        target = []
        for j, node in enumerate(model.graph.node):
            input_names = node.input
            for k, name in enumerate(input_names):
                if t.name == name:
                    share += 1
                    target.append((j, k))
        if share > 1:
            print('share %s %d times' % (t.name, share))
            for i in range(share):
                j, k = target[i]
                
                new_tensor = copy.deepcopy(t)
                new_name = t.name + f'_{i}'
                new_tensor.name = new_name

                model.graph.node[j].input[k] = new_name
                model.graph.initializer.append(new_tensor)
            tensor_to_del.append(t.name)
    # remove tensor
    print('remove tensor ', tensor_to_del)
    for name in tensor_to_del:
        for i, t in enumerate(model.graph.initializer):
            if t.name == name:
                print('remove initializer', model.graph.initializer[i].name)
                del model.graph.initializer[i]
        for i, t in enumerate(model.graph.node):
            if t.name == name:
                print('remove node', model.graph.node[i].name)
                del model.graph.node[i]
        for i, t in enumerate(model.graph.output):
            if t.name == name:
                print('remove output', model.graph.output[i].name)
                del model.graph.output[i]
        for i, t in enumerate(model.graph.input):
            if t.name == name:
                print('remove input', model.graph.input[i].name)
                del model.graph.input[i]

    onnx.save(model, onnx_file)

def convert(config_file, ckp_file, onnx_file, input_size=(224,224,3), device='cpu'):
    # Read mmdetection model
    mmdet_model = init_detector(config_file, ckp_file, device)
    # Init weights:
    if ckp_file is None:
        mmdet_model.init_weights()
    # Set forward function to dummy
    if hasattr(mmdet_model, 'forward_dummy'):
        mmdet_model.forward = mmdet_model.forward_dummy

    # Generate dummy data
    if len(input_size) == 0:
        input_size = (320,320,3)
    h, w, c = input_size
    x = torch.rand(1, c, h, w, device=torch.device(device))

    # Convert to onnx
    with torch.no_grad():
        torch.onnx.export(mmdet_model, x, onnx_file, verbose=False, input_names=['input'], opset_version=9) # 9

    # Deal with shared weights
    rename_share_weights(onnx_file)

    # Simplify nodes
    ox_model = onnx.load(onnx_file)
    model_simp, check = simplify(ox_model)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(model_simp, onnx_file)

    if isinstance(config_file, str):
        print('Convert {} to {} success'.format(config_file, onnx_file))
    else:
        print('Convert to {} success'.format(onnx_file))
    return (config_file, onnx_file)

def parse_input_size(args):
    if args.input_size is None:
        input_size = (320, 320, 3)
    else:
        input_size = tuple(args.input_size)
    return input_size

def main():
    args = parse_args()

    if args.debug:
        debug(args)

    input_size = parse_input_size(args)
    convert(args.config, args.checkpoint, args.onnx, input_size, args.device)

if __name__ == '__main__':
    main()