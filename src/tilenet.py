'''Modified ResNet-18 in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512,strat2=False, dictionary_labels=None, idx_include=None):
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64
        self.strat2 = strat2
        self.dictionary_labels = dictionary_labels 
        self.idx_include = idx_include
        
        
        if self.strat2: #if we weant to do strat2:
            self.secondary_classification = nn.Sequential(
                    nn.Linear(z_dim, 63) #66 = 
            )
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
      
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],
            stride=2)

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        return self.encode(x)

    def classification_loss(self, x, idx, loc): #softmax loss #x is 50 by 512
        score = self.secondary_classification(x) #output a 50 by 63 D vector
        score_upd = torch.sum(score,dim=0)
        #print(score_upd,score_upd.shape)
        label = self.dictionary_labels[idx][loc]
        softy = nn.Softmax()
        probs = softy(score_upd)
        #print(probs)
        loss = -math.log(probs[label])
        #print(loss)
        return loss
       
    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        #print(loss.type,loss)
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant,idx, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        big_loss,l_n,l_d,l_nd = self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)
        small_loss = 0
        if self.strat2:
            if idx in self.dictionary_labels:
                loss_z_p = self.classification_loss(z_p, idx, 0)
                #print(type(loss_z_p))
                loss_z_n = self.classification_loss(z_n, idx, 1)
                loss_z_d = self.classification_loss(z_d, idx, 2)
                small_loss += loss_z_p
                small_loss += loss_z_n
                small_loss += loss_z_d
        big_loss += small_loss
        return big_loss, small_loss, l_n,l_d,l_nd

def make_tilenet(in_channels=4, z_dim=512,strat2 = False, dictionary_labels=None, idx_include=None):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim,strat2=strat2 ,dictionary_labels=dictionary_labels, idx_include=idx_include)

