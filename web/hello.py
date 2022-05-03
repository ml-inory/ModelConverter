from flask import Flask, url_for, request, jsonify, redirect, send_file
from flask import render_template
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = './upload'
DOWNLOAD_FOLDER = './download'
TASK = {}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class TASK_CONFIG(object):
    def __init__(self):
        self.input_format = ''
        self.config = ''
        self.weight = ''
        self.output_format = ''

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return f'''input_format: {self.input_format}
        config: {self.config}
        weight: {self.weight}
        output_format: {self.output_format}
        '''

def convert_helper(username):
    input_format = TASK[username]['input_format']
    output_format = TASK[username]['output_format']
    config = TASK[username]['config']
    weight = TASK[username]['weight']

    if not os.path.exists(DOWNLOAD_FOLDER):
        os.mkdir(DOWNLOAD_FOLDER)

    user_path = os.path.join(DOWNLOAD_FOLDER, username)
    if not os.path.exists(user_path):
        os.mkdir(user_path)

    base_name = os.path.basename(weight).split('.')[0]
    onnx_name = base_name + '.onnx'
    onnx_path = os.path.join(user_path, onnx_name)
    if input_format == 'mmdet' or input_format == 'mmcls':
        exe = f'{input_format}2onnx'
        os.system(f'{exe} --config {config} --checkpoint {weight} --onnx {onnx_path}')
    else:
        os.system(f'python3 -m onnxsim {weight} {onnx_path}')

    if output_format == 'onnx':
        return onnx_name
    elif output_format == 'mnn':
        output_name = base_name + '.mnn'
        output_path = os.path.join(user_path, output_name)
        os.system(f'onnx2mnn --onnx {onnx_path} --mnn {output_path}')
    elif output_format == 'caffe':
        output_name = (base_name + ".prototxt", base_name + ".caffemodel")
        output_path = (os.path.join(user_path, i) for i in output_name)
        os.system(f'onnx2caffe --onnx {onnx_path} --prototxt {output_path[0]} --caffemodel {output_path[1]}')
    elif output_format == 'tflite':
        output_name = base_name + '.tflite'
        output_path = os.path.join(user_path, output_name)
        os.system(f'onnx2tflite --onnx {onnx_path} --tflite {output_path}')
    elif output_format == 'trt':
        output_name = base_name + '.trt'
        output_path = os.path.join(user_path, output_name)
        os.system(f'onnx2trt --onnx {onnx_path} --trt {output_path}')
    elif output_format == 'nnie':
        output_name = base_name + '.wk'
        print('Not supported for now!')
    return output_name
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        username = request.form.get('username', '')
        TASK[username] = TASK_CONFIG()
        print('Added user', username)
        return jsonify({'message': 'ok'})

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    if request.method == 'POST':
        username = request.form.get('username', '')
        input_format = request.form.get('input_format', '')
        output_format = request.form.get('output_format', '')

        TASK[username]['input_format'] = input_format
        TASK[username]['output_format'] = output_format

        print('username:', username)
        print(TASK[username])

        output_name = convert_helper(username)
        if not os.path.exists(os.path.join(DOWNLOAD_FOLDER, username, output_name)):
            return jsonify({'message': 'fail'})
        
        return jsonify({'message': 'ok', 'filename': output_name})

@app.route('/upload', methods=['POST'])
def upload():
    username = request.form.get('username', '')
    if username != '':
        if not os.path.exists(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        user_path = os.path.join(app.config['UPLOAD_FOLDER'], username)
        if not os.path.exists(user_path):
            os.mkdir(user_path)

        for file_type in ('config', 'weight'):
            if file_type in request.files:
                file = request.files[file_type]
                if file and file.filename != '':
                    filename = secure_filename(file.filename)
                    print(f'{file_type}:', filename)

                    save_path = os.path.join(user_path, filename)
                    TASK[username][file_type] = save_path
                    file.save(save_path)
                    return jsonify({'message': 'ok'})
                else:
                    return jsonify({'message': 'file not exist'})
    return jsonify({'message': 'no config or weight'})

@app.route('/download', methods=['GET'])
def download():
    if request.method == 'GET':
        username = request.args.get('username')
        filename = request.args.get('filename')
        print('username: ', request.args.get('username'))
        print('filename: ', request.args.get('filename'))
        print('Download: ', os.path.join(DOWNLOAD_FOLDER, username, filename))
        return send_file(safe_join(DOWNLOAD_FOLDER, username, filename), as_attachment=True)
        # return jsonify({'message': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 
