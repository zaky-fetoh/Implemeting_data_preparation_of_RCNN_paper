import torch.optim as optim
import torch.nn as nn
from model import *
import torch


# net= RCNN()
# net.cuda()
# closs = nn.CrossEntropyLoss()
# bloss = nn.L1Loss()
# opt = optim.Adam(net.parameters())



def save_model(net, ep_num,
               name='waight',
               outPath='./waights/'):
    file_name = outPath + name + str(ep_num) + '.pth'
    torch.save(net.state_dict(),
               file_name)
    print('Model Saved',file_name )

def load_model(file_path, model=RCNN()):
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    print('Model loaded', file_path )


def train(Rnet, tr_loader, va_loader, epochs,
          opt, cls_loss_fn, bb_loss_fn,
          device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu'),
    ):
    history= dict()
    for ep in range(epochs):
        history[ep] = list()
        for imgs, cls, bb in tr_loader:

            imgs,  bb = [x.to(device = device) for x in [imgs, bb]]
            cls = cls.to(device=device, dtype=torch.long).view(-1)

            pred_cls, pred_bb = Rnet(imgs)

            clss = cls_loss_fn(pred_cls, cls)
            blss = bb_loss_fn(pred_bb, bb)
            if
            gloss = clss + blss

            opt.zero_grad()
            gloss.backward()
            opt.step()
            print(gloss.item())

        save_model(Rnet, ep)





