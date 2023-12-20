import torch
import torch.nn as nn
from modules import *
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, block, blocks, c=3):
        super(Encoder, self).__init__()
        self.c = c
        self.in_planes, self.planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self.builder(block, self.planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self.builder(block, self.planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self.builder(block, self.planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self.builder(block, self.planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self.builder(block, self.planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.linear = nn.Sequential(nn.Linear(self.planes[4], self.planes[4]), nn.BatchNorm1d(self.planes[4]), nn.ReLU(inplace=True), nn.Linear(self.planes[4], self.planes[4]))

    def builder(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(DownSampler(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo, batch):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
            
        x5 = x5.reshape(batch, -1, self.planes[-1])
        x5 = torch.max(x5, dim=1)[0]
        
        return x5

class Decoder(nn.Module):
    def __init__(self, up_factors, dim_feat=512, num_pc=512, num_p0=512,
                 radius=1, bounding=True):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0           
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)              
        
        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial = None, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)

        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)     
        
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd

class ATT_Net(nn.Module):
    def __init__(self):
        super(ATT_Net, self).__init__()
        self.encoder = Encoder(FeatureExtractor, [2, 3, 4, 6, 3]).cuda()
        self.decoder = Decoder(dim_feat=512, num_pc=256, num_p0=512,
                               radius=1, bounding=True, up_factors=[4,1,4]).cuda()
        
    def forward(self, partial, batch_size):
        pp, xp, op = partial.pos, partial.pos, partial.ptr[1:] 
        
        encoding = self.encoder([pp, xp, op], batch_size)
        complete = self.decoder(encoding.unsqueeze(2), partial = partial.pos.reshape(batch_size,-1,3))[-1]
        
        return complete
            
