import torch.nn as nn
import model as cmodel
import data_org as dorg
import numpy as np
import torch
import torchvision.transforms as transf
import torch.optim as optim
import cv2 as cv

transform = transf.Compose([
                     transf.ToPILImage(),
                     transf.ToTensor(),
                     transf.Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225)),
                 ])
def serve(net, image):
    regions = dorg.selective_Search(image)
    output = list()
    for bb in regions:
        imbx = dorg.crop_image(image, bb)
        timbx = cv.resize(imbx, (224,224))
        timbx = transform(timbx).cuda()
        cls_pred, bb_pred = net(torch.unsqueeze(timbx, 0))

        cls_pred = cls_pred.detach().cpu().argmax(1).view(-1)[0]
        bb_pred = bb_pred.detach().cpu().numpy()
        correctedbb =  np.array(bb) + bb_pred
        correctedbb = correctedbb.tolist()
        print(correctedbb + [cls_pred])
        if cls_pred :
            output.append([cls_pred]+correctedbb)
    return output











if __name__ == '__main__':
    dorg.download_and_extract()
    dorg.show_the_resulted_dataset_after_crops()
    #bloss = nn.L1Loss()
    #rnet = cmodel.RCNN().cuda()

    #cmodel.save_model(rnet, 3)

    # closs = nn.CrossEntropyLoss()
    # trldr, valdr = dorg.getloaders_dff()
    # opt = optim.Adam(rnet.parameters(),1e-4)

    #cmodel.train(rnet,trldr,valdr,30, opt,closs, bloss)

    # img = cv.imread('tes2.jpg')
    # out = serve(rnet, img)
    #
    # dorg.image_with_BB(img, out)
    




