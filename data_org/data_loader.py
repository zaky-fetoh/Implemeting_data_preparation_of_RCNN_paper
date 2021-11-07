import torch.utils.data as udate
from data_org import *

def getloaders_dff(train_set_path = TRAIN_SET_LABELS,
               val_set_path = VAL_SET_LABELS,
               ):
    trldr = getloader_train(train_set_path)
    valdr = getloader_train(val_set_path)
    return trldr, valdr


def getloader_train(train_path = TRAIN_SET_LABELS,
                    batch_size = 64,
                    batch_per_image=64, positive_patches=32,
                    ):
    dts = RCNN_ready_dataset(train_path,batch_per_image,
                       positive_patches,
                       )
    tdlr = udate.DataLoader(dts,batch_size,#shuffle=True,
                            #pin_memory=True, num_workers=2
                            )
    return tdlr
