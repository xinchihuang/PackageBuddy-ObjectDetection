import os
os.system("pip install numpy cython scikit_image keras==2.1.6 h5py imgaug opencv_python")
# os.system("pip uninstall pycocotools")
os.system("pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI")
cwd=os.getcwd()
os.system("(cd "+cwd+"lib/Mask_RCNN/samples/coco/PythonAPI && make)")