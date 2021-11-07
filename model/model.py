import torchvision.models as tvmodels
import torch.nn as nn
import torch

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    def forward(self, x):
        return x
class dff_backbone(nn.Module):
    def __init__(self):
        super(dff_backbone, self).__init__()
        self.resnet = tvmodels.resnet34(True)
        #self.resnet.fc = identity()
    def forward(self, X):
        return self.resnet(X)
class cls_head(nn.Module):
    def __init__(self,inputlen=1000, num_out_cls= 21):
        super(cls_head, self).__init__()
        self.layers = nn.Sequential(
            #nn.Linear(inputlen,64),#clsHead
            nn.ReLU(),
            nn.Linear(inputlen,num_out_cls),
        )
    def forward(self, x):
        return self.layers(x)

class bb_head( nn.Module):
    def __init__(self, inputlen= 1000):
        super(bb_head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputlen, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            #nn.ReLU(),
            #nn.Linear(64,4),
        )
    def forward(self, x):
        return self.layers(x)

class RCNN( nn.Module):
    def __init__(self, backbone = dff_backbone(),
                 cls_head_nn=cls_head(), bb_head_nn= bb_head()):
        super(RCNN, self).__init__()
        self.backbone = backbone
        self.cls_head = cls_head_nn
        self.bb_head =bb_head_nn
    def forward(self, x):
        intrep = self.backbone(x)
        return self.cls_head(intrep), self.bb_head(intrep)



if __name__ == '__main__':
    tens = torch.rand(3,3,224,224)
    net = RCNN()
    out = net(tens)
    print( out)