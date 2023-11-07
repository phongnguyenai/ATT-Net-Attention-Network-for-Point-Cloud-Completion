conda create -n att_env python=3.10
conda activate att_env
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
cd Pointnet2_Pytorch
python setup.py install
