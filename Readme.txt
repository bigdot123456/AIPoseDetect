1. python环境配置
    - python3.6,最好用conda维护python环境
    - tensorflow 1.4.1+   cv2
    - requirement.txt中的python包（pip3 install -r requirements.txt）
    - 编译后处理的 c++ library，执行：
        cd tf_pose/pafprocess
        sudo apt install swig
        swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
    - 安装工程：
        cd tf-openpose
        python setup.py install
    - 测试安装是否成功，执行：
        python -c 'import tf_pose; tf_pose.infer(image="./images/p1.jpg")'

备注：工程安装好之后，即可调用图像处理函数，得到结果。

2.函数调用方法
    - 说明：输入