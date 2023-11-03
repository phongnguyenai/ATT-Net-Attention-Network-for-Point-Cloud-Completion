from __future__ import print_function
import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model import ATT_Net

import sys
sys.path.append("../../IEEE-Sensor-2023/code/")

from datasets import PCNDataset
from pytorch3d.loss.chamfer import chamfer_distance
import argparse
import sys
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train or test", default="train")
parser.add_argument("--pretrained", help="pth file", default="")
parser.add_argument("--car", help="True or False", default=True)
parser.add_argument("--batch-size", help="Batch size", default=8)
parser.add_argument("--model-name", help="Name of the method", default="ATT-Net")
parser.add_argument("--epoch", help="Epoch", default=401)
parser.add_argument("--num-pred", help="Number of output point cloud", default=8192)

args = parser.parse_args()

class Config( object ):
    def __init__( self ):
        self.myAttr= None

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch):
    model.train()
    train_loss = 0
    count = 0
    for data in train_dataloader:
        gt = data[3].to(device)
        partial = data[2].to(device)
        batch = partial.pos.reshape(-1, 2048, 3).shape[0]
        
        # Prediction
        optimizer.zero_grad()
        
        pc = model(partial, batch)
        loss = chamfer_distance(pc, gt.pos.reshape(batch,-1,3), norm = 2)[0]*1000
        
        torch.cuda.empty_cache()
        loss.backward()
        train_loss += loss.item() * batch
        optimizer.step()
        count +=1
        print(f"{model_name} - Training epoch {epoch}: {int(count/len(train_dataloader)*100)}%", end='\r')
        sys.stdout.flush()
        
    return train_loss / len(train_dataset) / 1000

def evaluation():
    model.eval()
    val_loss = 0
    
    for data in val_dataloader:      
        gt = data[3].to(device)
        partial = data[2].to(device)
        batch = partial.pos.reshape(-1, 2048, 3).shape[0]
        
        # Prediction
        with torch.no_grad():
            pc = model(partial, batch)
            chamfer_loss = chamfer_distance(pc, gt.pos.reshape(batch,-1,3), norm = 2)[0]
            val_loss += chamfer_loss.item() * batch
            
    return val_loss/len(val_dataset)

def test():
    model.eval()
    test_loss = 0
    
    for data in test_dataloader:      
        gt = data[3].to(device)
        partial = data[2].to(device)
        batch = partial.pos.reshape(-1, 2048, 3).shape[0]
        
        # Prediction
        with torch.no_grad():
            pc = model(partial, batch)
            chamfer_loss = chamfer_distance(pc, gt.pos.reshape(batch,-1,3), norm = 2)[0]
            test_loss += chamfer_loss.item() * batch
            
    return test_loss/len(test_dataset)

seed_everything()

# Train cofig
train_config = Config()
train_config.subset = "train"
train_config.PARTIAL_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{train_config.subset}/partial"
train_config.COMPLETE_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{train_config.subset}/complete"
train_config.CATEGORY_FILE_PATH = "../../IEEE-Sensor-2023/code/PCN/PCN.json"
train_config.N_POINTS = int(args.num_pred)
train_config.CARS = False

# Valid cofig
val_config = Config()
val_config.subset = "val"
val_config.PARTIAL_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{val_config.subset}/partial"
val_config.COMPLETE_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{val_config.subset}/complete"
val_config.CATEGORY_FILE_PATH = "../../IEEE-Sensor-2023/code/PCN/PCN.json"
val_config.N_POINTS = int(args.num_pred)
val_config.CARS = False

# Test cofig
test_config = Config()
test_config.subset = "test"
test_config.PARTIAL_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{test_config.subset}/partial"
test_config.COMPLETE_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{test_config.subset}/complete"
test_config.CATEGORY_FILE_PATH = "../../IEEE-Sensor-2023/code/PCN/PCN.json"
test_config.N_POINTS = int(args.num_pred)
test_config.CARS = False

# Dataset
train_dataset = PCNDataset.PCN(train_config)
val_dataset = PCNDataset.PCN(val_config)
test_dataset = PCNDataset.PCN(test_config)

# Dataloader
batch_size = int(args.batch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                    shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                    shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                    shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ATT_Net().to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00007)

# Model name
model_name = args.model_name

# torch.cuda.empty_cache()
# torch.set_num_threads(24)

min_loss = 99999999999999

path = f"weights/{model_name}"
if not os.path.exists(path):
    os.makedirs(path)
    
if args.mode == "train":
    scheduler = StepLR(optimizer, step_size=40, gamma=0.7)
    # Start training
    for epoch in range(int(args.epoch)):
        train_loss = train(epoch)
        eval_loss = evaluation()
            
        f = open(f'{path}/log_loss.txt', 'a')
        str_loss = "Epoch {:03d} - Training loss: {:.10f} - Validation loss: {:.10f} \n".format(epoch, train_loss, eval_loss)
        f.write(str_loss)
        print(str_loss)

        if eval_loss < min_loss:
            torch.save(model.state_dict(),f'./{path}/best.pt')
            min_loss = eval_loss

        torch.save(model.state_dict(),f'./{path}/last.pt')
        scheduler.step()

    # Print results
    torch.cuda.empty_cache()
    model = ATT_Net().to(device)
    model.load_state_dict(torch.load(f"{path}/best.pt"))
    test_loss = test()

    str_test = "Test loss: {:.10f}".format(test_loss)
    f.write(str_test)
    print(str_test)

    
