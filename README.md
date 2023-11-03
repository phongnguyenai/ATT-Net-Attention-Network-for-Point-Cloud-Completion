# ATT-Net: Attention Network for Point Cloud Completion

## Installation

Before you begin, ensure you have the following prerequisites installed:

- **CUDA**: Version 11.8
- **PyTorch**: Version 2.0.1 and **torch_geometric**: Version 2.3.1. You can install them from [pytorch.org](https://pytorch.org/).
- **pytorch3d**: Version 0.7.4. Install it by following the instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
- **pointnet2_ops**: To install locally, follow these steps:

    ```bash
    cd Pointnet2_Pytorch
    python setup.py install
    ```

## Prediction

To make predictions, follow these steps:

1. Download the test dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/dataset/ATT-Net/test.tar.gz?csf=1&web=1&e=Sn6rpK).

2. Extract the downloaded file and copy its contents to the folder `./PCN/test/`.

3. Run the following command to perform predictions:

    ```bash
    python test.py
    ```

## Training

To train the model, you'll need to follow these steps:

1. Download the validation dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EbxYcKtV_ahOpaAvq-A-9ZwBOqabr_5nddl7mWwhWJJ_Rw?e=FSiE7A).

2. Download the training dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EeffEPj7HgpGhkGQVshxqWwBRz6bGUjLmirj79GgFflyCA?e=HhemQE).

3. After downloading, extract the validation and training dataset files.

4. Copy the extracted files to the folders `./PCN/train/` and `./PCN/val`.

5. To initiate the training process, execute the following command:

    ```bash
    python train.py
    ```

This organized format should help users better understand and follow the installation, prediction, and training steps for your project.
