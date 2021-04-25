# -*- coding: utf-8 -*-
import os, sys
import glob
import numpy as np
sys.path.append('/usr/local/python')
os.environ['GLOG_minloglevel'] = '3'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
caffe.set_mode_cpu()

# __all__ = ['convert', 'check']

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Caffe to NNIE model Converter')
    parser.add_argument('--output', help='generated model wk', required=True)
    parser.add_argument('--prototxt', help='caffe prototxt path', required=False)
    parser.add_argument('--caffemodel', help='caffe weight path', required=True)
    parser.add_argument('--image_dir', help='Image folder used for quantizing', required=True)
    parser.add_argument('--RGB', help='preprocess param: convert to RGB, default False', default=False, action='store_true')
    parser.add_argument('--scale', help='preprocess param: img = img * scale', default=1.0, type=float)
    parser.add_argument('--mean', help='preprocess param: img = img - mean', default=0, type=float, nargs='*')
    
    args = parser.parse_args()
    return args


def write_f(f, key, value):
    f.write(f'[{key}] {value}\n')

def gen_imageList(image_dir):
    with open('imageList.txt', 'w') as f:
        imgs = glob.glob(image_dir + '/*.jpg')
        for img in imgs:
            f.write(img + '\n')

def gen_mean(mean):
    with open('mean.txt', 'w') as f:
        if len(mean) == 1:
            f.write(f'{mean[0]}\n{mean[0]}\n{mean[0]}')
        else:
            f.write(f'{mean[0]}\n{mean[1]}\n{mean[2]}')

def convert(output, prototxt, caffemodel, image_dir, work_dir='../../mapper/', RGB=True, preprocess=True, scale=0.0078125, mean=[127.5], int8=True):
    cur_path = os.path.abspath(os.getcwd())

    os.chdir(work_dir)
    gen_imageList(os.path.join(cur_path, image_dir))

    model_name = os.path.splitext(os.path.basename(output))[0]
    cfg_file = model_name + '.cfg'
    with open(cfg_file, 'w') as f:
        write_f(f, 'prototxt_file', os.path.join(cur_path, prototxt))
        write_f(f, 'caffemodel_file', os.path.join(cur_path, caffemodel))
        write_f(f, 'net_type', 0)
        write_f(f, 'image_list', './imageList.txt')
        write_f(f, 'image_type', 1)
        write_f(f, 'instruction_name', model_name)

        if preprocess:
            write_f(f, 'norm_type', 5)
            write_f(f, 'data_scale', scale)
            gen_mean(mean)
            write_f(f, 'mean_file', './mean.txt')
        else:
            write_f(f, 'norm_type', 0)
            write_f(f, 'data_scale', '1.0')
            write_f(f, 'mean_file', 'null')


        if int8:
            write_f(f, 'compile_mode', 0)
        else:
            write_f(f, 'compile_mode', 1)

        if RGB:
            write_f(f, 'RGB_order', 'RGB')
        else:
            write_f(f, 'RGB_order', 'BGR')

    print(f'Generate {cfg_file}')
    
    # remember LD_LIBRARY_PATH
    tmp_ld = os.environ['LD_LIBRARY_PATH']
    # os.system('bash setup.sh')
    os.environ['LD_LIBRARY_PATH'] += f':{work_dir}'
    # print(os.environ['LD_LIBRARY_PATH'])
    os.system(f'./nnie_mapper_12 {cfg_file}')
    os.chdir(cur_path)
    os.system(f'mv {work_dir}/{model_name}.wk {output}')
    os.environ['LD_LIBRARY_PATH'] = tmp_ld

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

def auto_depthwise(prototxt_path):
    print('\n======== optimize: auto depthwise ========')
    net_specs = caffe_pb2.NetParameter()
    with open(prototxt_path, "r") as fp:
        text_format.Merge(str(fp.read()), net_specs)

    layers = net_specs.layer
    num_layers = len(layers)

    # find all conv
    convs = []
    for i in range(num_layers):
        if layers[i].type == 'Convolution':
            convs.append(i)

    for i in convs:
        num_output = layers[i].convolution_param.num_output
        group = layers[i].convolution_param.group
        if num_output == group:
            print('Change %s to DepthwiseConv' % layers[i].name)
            layers[i].type = 'DepthwiseConv'
            layers[i].convolution_param.group = 1

    net_str = '{}'.format(net_specs)
    net_str = net_str.replace('group: 1', '# group: 1')

    with open(prototxt_path, "w") as fp:
        fp.write(net_str)

def fix_axis(prototxt_path):
    print('\n======== optimize: fix axis ========')
    net_specs = caffe_pb2.NetParameter()
    with open(prototxt_path, "r") as fp:
        text_format.Merge(str(fp.read()), net_specs)

    layers = net_specs.layer
    num_layers = len(layers)

    # find all Concat
    concats = []
    for i in range(num_layers):
        if layers[i].type == 'Concat':
            concats.append(i)

    # find all Reshape before Concat
    reshapes = []
    need_to_fix = [False]*len(concats)
    for i in concats:
        bottoms = layers[i].bottom
        reshapes.append([])
        for k in range(num_layers):
            if layers[k].type == 'Reshape' and layers[k].top[0] in bottoms:
                reshapes[-1].append(k)

    # find Softmax after Concat
    softmaxs = []
    for i in concats:
        top = layers[i].top[0]
        for k in range(num_layers):
            if layers[k].type == 'Softmax' and layers[k].bottom[0] == top:
                softmaxs.append(k)

    # fix Reshape
    if len(reshapes) > 0:
        for i, reshape in enumerate(reshapes):
            if len(reshape) > 0 and len(layers[reshape[0]].reshape_param.shape.dim) == 3:
                need_to_fix[i] = True
            for k in reshape:
                if len(layers[k].reshape_param.shape.dim) == 3:
                    print('fix Reshape param of %s' % layers[k].name)
                    layers[k].reshape_param.shape.dim.insert(1, 1)

    # fix Concat
    for i, b in enumerate(need_to_fix):
        if b:
            print('fix Concat param of %s' % layers[concats[i]].name)
            layers[concats[i]].concat_param.axis = 2

    # fix Softmax
    for i in softmaxs:
        if layers[i].softmax_param.axis == 2:
            print('fix Softmax param of %s' % layers[i].name)
            layers[i].softmax_param.axis = 3

    with open(prototxt_path, "w") as fp:
        fp.write("{}".format(net_specs))



def optimize(old_ptx, old_model, new_ptx, new_model):
    print('Optimize caffe model')
    merge_bn(old_ptx, old_model, new_ptx, new_model)
    auto_inplace(new_ptx)
    auto_depthwise(new_ptx)
    fix_axis(new_ptx)

def main():
    if 'NNIE_MAPPER' not in os.environ.keys():
        print('Please export NNIE_MAPPER first! Set it to the path of nnie_mapper')
        sys.exit(-1)

    args = parse_args()
    work_dir = os.environ['NNIE_MAPPER']
    old_ptx = args.prototxt
    old_model = args.caffemodel
    new_ptx = old_ptx.replace('.prototxt', '_optim.prototxt')
    new_model = old_model.replace('.caffemodel', '_optim.caffemodel')
    optimize(old_ptx, old_model, new_ptx, new_model)
    convert(args.output, new_ptx, new_model, args.image_dir, work_dir, args.RGB, True, args.scale, args.mean, True)

if __name__ == '__main__':
    main()