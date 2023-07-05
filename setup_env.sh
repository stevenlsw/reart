pip install -r requirements.txt
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
python setup.py install
cd ..
rm -r chamferdist
cd networks/pointnet_lib
python setup.py install # Compile the CUDA code for PointNet++ backbone
