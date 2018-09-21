#### python环境配置
1. python3.6,最好用conda维护python环境，安装anaconda
2. 安装tensorflow 1.4.1+， cv2
```
清华镜像站下载anaconda安装包安装,参考 https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
source ~/.barshrc
conda create -n tensorflow python=3.6
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl
conda install --channel https://conda.anaconda.org/menpo opencv3
pip install opencv-python
```
3. 安装requirement.txt中的python包，执行：
```
        pip3 install -r requirements.txt
```
4. 编译后处理的 c++ library，执行：
```
        cd tf_pose/pafprocess
        sudo apt install swig
        swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```
5. 安装工程：
```
        cd tf-openpose
        python setup.py install
```
6. 测试安装是否成功，执行：
```
    python -c 'import tf_pose; tf_pose.infer(image="./images/p1.jpg")'
```
7. 得到如下图说明成功：

![](http://ww1.sinaimg.cn/large/8833244fgy1ft2cxsstvbj206106g3yu.jpg)

8. 备注：工程安装好之后，即可调用图像处理函数，得到结果。

#### 函数调用方法
```
输入参数：
    path： 需要处理的图片路径， 比如：path='./images/pi.jpg'
    
返回：
    image：处理之后得到的图像结果， opencv图像格式


from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def process(path):
    image = common.read_imgfile(path=path, width=432, height=368)
    e = TfPoseEstimator(get_graph_path(model_name='mobilenet_thin'), target_size=(432, 368))
    humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    return image
```