from __future__ import print_function
from __future__ import absolute_import

import os
import sys
sys.path.append('/usr/local/python')
os.environ['GLOG_minloglevel'] = '3'
import caffe
import onnx
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
caffe.set_mode_cpu()

from .onnx2caffe_src._transformers import ConvAddFuser,ConstantsToInitializers
from .onnx2caffe_src._graph import Graph

from .onnx2caffe_src._operators import make_input, _ONNX_NODE_REGISTRY as cvt_ONNX_NODE_REGISTRY
from .onnx2caffe_src._weightloader import _ONNX_NODE_REGISTRY as wlr_ONNX_NODE_REGISTRY
# import onnx2caffe_src._operators as cvt
# import onnx2caffe_src._weightloader as wlr
from .onnx2caffe_src._error_utils import ErrorHandling
from collections import OrderedDict
from onnx import shape_inference
import importlib


transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ONNX to Caffe model Converter')
    parser.add_argument('--onnx', help='onnx model file path', required=True)
    parser.add_argument('--prototxt', help='caffe prototxt path', required=False)
    parser.add_argument('--caffemodel', help='caffe weight path', required=True)
    parser.add_argument('--use_external_prototxt', help='whether use external prototxt, default False', default=False, action='store_true')
    
    args = parser.parse_args()
    return args

def convertToCaffe(graph, prototxt_save_path, caffe_model_save_path, save_prototxt=True):
    # print('convert to caffe')
    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    for i in graph.inputs:
        edge_name = i[0]
        # input_layer = cvt.make_input(i)
        input_layer = make_input(i)
        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]


    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False

        # print(f'node_name: {node_name}  op_type: {op_type}')

        for inp in inputs:
            if inp not in exist_edges and inp not in inputs_tensor:
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        # if op_type not in cvt._ONNX_NODE_REGISTRY:
        if op_type not in cvt_ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        # converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
        converter_fn = cvt_ONNX_NODE_REGISTRY[op_type]
        layer = converter_fn(node,graph,err)
        if type(layer)==tuple:
            for l in layer:
                layers.append(l)
        else:
            layers.append(layer)
        outs = node.outputs
        for out in outs:
            exist_edges.append(out)

    net = caffe_pb2.NetParameter()
    for id,layer in enumerate(layers):
        layers[id] = layer._to_proto()
    net.layer.extend(layers)

    if save_prototxt:
        with open(prototxt_save_path, 'w') as f:
            print(net,file=f)

    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        # if op_type not in wlr._ONNX_NODE_REGISTRY:
        if op_type not in wlr_ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        # converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        converter_fn = wlr_ONNX_NODE_REGISTRY[op_type]
        converter_fn(net, node, graph, err)

    net.save(caffe_model_save_path)
    return net

def getGraph(onnx_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph

def convert(onnx_path, prototxt_path, caffemodel_path, save_prototxt=True):
    graph = getGraph(onnx_path)
    convertToCaffe(graph, prototxt_path, caffemodel_path, save_prototxt)

bn_maps = {}
def find_top_after_bn(layers, name, top):
    bn_maps[name] = {} 
    for l in layers:
        if len(l.bottom) == 0:
            continue
        if l.bottom[0] == top and l.type == "BatchNorm":
            bn_maps[name]["bn"] = l.name
            top = l.top[0]
        if l.bottom[0] == top and l.type == "Scale":
            bn_maps[name]["scale"] = l.name
            top = l.top[0]
    return top

def pre_process(expected_proto, new_proto):
    net_specs = caffe_pb2.NetParameter()
    net_specs2 = caffe_pb2.NetParameter()
    with open(expected_proto, "r") as fp:
        text_format.Merge(str(fp.read()), net_specs)

    net_specs2.MergeFrom(net_specs)
    layers = net_specs.layer
    num_layers = len(layers)

    for i in range(num_layers - 1, -1, -1):
         del net_specs2.layer[i]

    for idx in range(num_layers):
        l = layers[idx]
        if l.type == "BatchNorm" or l.type == "Scale":
            continue
        elif l.type == "Convolution" or l.type == "Deconvolution":
            top = find_top_after_bn(layers, l.name, l.top[0])
            bn_maps[l.name]["type"] = l.type
            layer = net_specs2.layer.add()
            layer.MergeFrom(l)
            layer.top[0] = top
            layer.convolution_param.bias_term = True
        else:
            layer = net_specs2.layer.add()
            layer.MergeFrom(l)

    with open(new_proto, "w") as fp:
        fp.write("{}".format(net_specs2))

def load_weights(net, nobn):
    if sys.version_info > (3,0):
        listKeys = nobn.params.keys()
    else:
        listKeys = nobn.params.iterkeys()
    for key in listKeys:
        if type(nobn.params[key]) is caffe._caffe.BlobVec:
            conv = net.params[key]
            if key not in bn_maps or "bn" not in bn_maps[key]:
                for i, w in enumerate(conv):
                    nobn.params[key][i].data[...] = w.data
            else:
                bn = net.params[bn_maps[key]["bn"]]
                scale = net.params[bn_maps[key]["scale"]]
                wt = conv[0].data
                channels = 0
                if bn_maps[key]["type"] == "Convolution": 
                    channels = wt.shape[0]
                elif bn_maps[key]["type"] == "Deconvolution": 
                    channels = wt.shape[1]
                else:
                    print("error type " + bn_maps[key]["type"])
                    exit(-1)
                bias = np.zeros(channels)
                if len(conv) > 1:
                    bias = conv[1].data
                mean = bn[0].data
                var = bn[1].data
                scalef = bn[2].data

                scales = scale[0].data
                shift = scale[1].data

                if scalef != 0:
                    scalef = 1. / scalef
                mean = mean * scalef
                var = var * scalef
                rstd = 1. / np.sqrt(var + 1e-5)
                if bn_maps[key]["type"] == "Convolution": 
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                else:
                    rstd1 = rstd.reshape((1, channels,1,1))
                    scales1 = scales.reshape((1, channels,1,1))
                    wt = wt * rstd1 * scales1
                bias = (bias - mean) * rstd * scales + shift
                
                nobn.params[key][0].data[...] = wt
                nobn.params[key][1].data[...] = bias

def merge_bn(prototxt_path, caffemodel_path):
    print('======== optimize: merge BN ========')
    pre_process(prototxt_path, "no_bn.prototxt")
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)  
    net2 = caffe.Net("no_bn.prototxt", caffe.TEST)
    load_weights(net, net2)
    net2.save("no_bn.caffemodel")
    os.system(f'mv no_bn.prototxt {prototxt_path}')
    os.system(f'mv no_bn.caffemodel {caffemodel_path}')

def auto_inplace(prototxt_path):
    print('======== optimize: auto inplace ========')
    net_specs = caffe_pb2.NetParameter()
    with open(prototxt_path, "r") as fp:
        text_format.Merge(str(fp.read()), net_specs)

    layers = net_specs.layer
    num_layers = len(layers)
    print('num layers', num_layers)

    # find all conv
    convs = []
    for i in range(num_layers):
        if layers[i].type == 'Convolution':
            convs.append(i)
    # find all uninplace relu after conv
    relus = []
    changed_names = [] # (from, to)
    for i in convs:
        top = layers[i].top[0]
        for k in range(num_layers):
            if len(layers[k].bottom) > 0 and layers[k].bottom[0] == top and layers[k].type in ('ReLU','RelU6','LeakyReLU','PReLU') and layers[k].top[0] != layers[k].bottom[0]:
                relus.append(k)
                changed_names.append((layers[k].top[0], top))
    # inplace relu
    for i in relus:
        layers[i].top[0] = layers[i].bottom[0]
    # change other connected bottom
    for old_bottom, new_bottom in changed_names:
        for i in range(num_layers):
            num_bottom = len(layers[i].bottom)
            for k in range(num_bottom):
                if layers[i].bottom[k] == old_bottom:
                    layers[i].bottom[k] = new_bottom
    
    with open(prototxt_path, "w") as fp:
        fp.write("{}".format(net_specs))


def optimize(prototxt_path, caffemodel_path):
    print('Optimize caffe model')
    merge_bn(prototxt_path, caffemodel_path)
    auto_inplace(prototxt_path)
    

def main():
    args = parse_args()
    convert(args.onnx, args.prototxt, args.caffemodel, not args.use_external_prototxt)
    optimize(args.prototxt, args.caffemodel)

if __name__ == '__main__':
    main()

