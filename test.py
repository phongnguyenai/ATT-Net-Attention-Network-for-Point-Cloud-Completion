from __future__ import print_function
import os
import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from model import ATT_Net

import sys
sys.path.append("../../IEEE-Sensor-2023/code/")

from datasets import PCNDataset
from pytorch3d.loss.chamfer import chamfer_distance
import sys

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

def test():
    print("Test in progress...")
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

# Test cofig
test_config = Config()
test_config.subset = "test"
test_config.PARTIAL_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{test_config.subset}/partial"
test_config.COMPLETE_POINTS_PATH = f"../../IEEE-Sensor-2023/code/PCN/{test_config.subset}/complete"
test_config.CATEGORY_FILE_PATH = "../../IEEE-Sensor-2023/code/PCN/PCN.json"
test_config.N_POINTS = 8192
test_config.CARS = False

test_dataset = PCNDataset.PCN(test_config)

test_dataloader = DataLoader(test_dataset, batch_size=8,
                    shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ATT_Net().to(device)
model.load_state_dict(torch.load(f"weights/ATT-Net/best.pt"))
test_loss = test()

str_test = "Chamfer Distance: {:.10f}".format(test_loss)
print(str_test)