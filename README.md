# ATT-Net-Attention-Network-for-Point-Cloud-Completion

1. Installation

- CUDA 11.8
- torch==2.0.1, torch_geometric==2.3.1. Install at https://pytorch.org/
- pytorch3d==0.7.4. Install at https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- pointnet2_ops. Install locally as follows:

```
cd Pointnet2_Pytorch
python setup.py install
```

2. Prediction

Download test dataset at: https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/dataset/ATT-Net/test.tar.gz?csf=1&web=1&e=Sn6rpK

Extract the file and copy it to the folder ./PCN/test/

Run this command:

```
python test.py
```

3. Training

Download val dataset at: https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EbxYcKtV_ahOpaAvq-A-9ZwBOqabr_5nddl7mWwhWJJ_Rw?e=FSiE7A

Download train dataset at: 

Extract val and train files. Then copy them to the folder ./PCN/train/ and ./PCN/val

Run this command:

```
python train.py
```