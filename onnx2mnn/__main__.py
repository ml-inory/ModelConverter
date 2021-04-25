# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse

# ONNX
import onnx
from onnxsim import simplify
import onnxruntime
import copy

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ONNX to MNN model Converter')
    parser.add_argument('--onnx', help='onnx model file path', required=True)
    parser.add_argument('--mnn', help='mnn model file path', required=True)
    args = parser.parse_args()
    return args

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
    if len(tensor_to_del) > 0:
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

def fix_resize_nodes(onnx_file):
    from onnx import helper
    model = onnx.load(onnx_file)
    tensor_to_del = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Resize':
            input_num = len(node.input)
            if input_num == 3:
                print(f'Resize op {node.name} must have 2 or 4 inputs, remove roi {node.input[1]}')
                tensor_to_del.append(node.input[1])

                upsample_node = helper.make_node('Upsample', [node.input[0], node.input[2]], node.output, name=node.name)
                model.graph.node[i].CopyFrom(upsample_node)

    for name in tensor_to_del:
        for i, t in enumerate(model.graph.node):
            if t.op_type == 'Constant' and t.output[0] == name:
                print('remove node', t.name)
                del model.graph.node[i]

    onnx.save(model, onnx_file)

def convert(onnx_file, mnn_file):
    # Deal with shared weights
    rename_share_weights(onnx_file)
    # Deal with resize node
    # fix_resize_nodes(onnx_file)

    os.system(f'MNNConvert -f ONNX --modelFile {onnx_file} --MNNModel {mnn_file} --bizCode none')
    
def main():
    args = parse_args()
    convert(args.onnx, args.mnn)

if __name__ == '__main__':
    main()