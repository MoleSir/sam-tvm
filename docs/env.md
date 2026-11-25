git clone https://github.com/facebookresearch/segment-anything.git 
git clone https://gitee.com/MoleSir/segment-anything.git

conda create -n sam_tvm  python=3.10 -y
conda activate sam_tvm
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./segment-anything
pip install opencv-python pycocotools matplotlib onnx onnxruntime
pip install apache-tvm
pip install cuda-bindings