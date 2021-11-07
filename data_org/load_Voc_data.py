import torchvision.transforms as transf
import torch.utils.data as tdata
from data_org import *
import cv2 as cv
import torch


# a dataset object that read a raw images without further processing
class Raw_Voc_dataset(tdata.Dataset):
    def __init__(self, label_file_path):
        self.images_ids = get_ids(label_file_path)
    def __len__(self):
        return self.images_ids.__len__()
    def __getitem__(self, index):
        img = get_image(self.images_ids[index])
        mdt = get_cls_bb(self.images_ids[index])
        return img, mdt


# a nextlayer above Raw_Voc_dataset such that it process the raw image
# to batchs of positive and negative patchs of the original images
# batch is contextly defined as crop of the image
# each image of the raw image is further divided to N image
class RCNN_custum_dataset(tdata.Dataset):

    def __init__(self, label_file_path, batchs_per_image=128,
                 positive_patchs=32):
        self.raw_images = Raw_Voc_dataset(label_file_path)
        self.batchs_per_image = batchs_per_image
        self.positive_patchs = positive_patchs
        self.negative_patchs = batchs_per_image - positive_patchs

        self.current_mdt =None
        self.current_batchs = list()
        self.current_image_index = None
        self.current_img = None
        self.current_reff = None
    def __len__(self):
        return self.raw_images.__len__() * self.batchs_per_image

    def __getitem__(self, index):
        raw_index = index // self.batchs_per_image
        batch_index = index % self.batchs_per_image
        if raw_index != self.current_image_index:
            self.current_image_index = raw_index
            img, mdt = self.raw_images.__getitem__(raw_index)
            self.current_img = img
            self.current_mdt = mdt
            rects = selective_Search(img)
            po, ne, poreff, nereff = selective_out_clustering(rects, mdt['object'][:])
            self.current_batchs = po[:self.positive_patchs] + ne[:self.negative_patchs]
            self.current_reff = poreff[:self.positive_patchs] + nereff[:self.negative_patchs]

        clbb = self.current_batchs[batch_index % self.current_batchs.__len__()]
        reff = self.current_reff[batch_index % self.current_batchs.__len__()]
        return crop_image(self.current_img, clbb[1:]), clbb[0], reff


class RCNN_ready_dataset(tdata.Dataset):
    def __init__(self, label_file_path, batchs_per_image=128,
                 positive_patchs=32, im_resize=(224,224),
                 transforms=transf.Compose([
                     transf.ToPILImage(),
                     # transf.RandomHorizontalFlip(),
                     # transf.RandomVerticalFlip(),
                     # transf.RandomRotation(20),
                     transf.ToTensor(),
                     # transf.Normalize((0.485, 0.456, 0.406),
                     #                  (0.229, 0.224, 0.225)),
                 ])):
        self.resize_shape = im_resize
        self.dataset = RCNN_custum_dataset(label_file_path,
                                           batchs_per_image,
                                           positive_patchs)
        self.transforms = transforms

    def __len__(self):
        return self.dataset.__len__()
    def __getitem__(self, item):
        img, cls, rebb =  self.dataset.__getitem__(item)
        img = cv.resize(img, self.resize_shape)
        img = self.transforms(img).view(3,*self.resize_shape)
        cls = torch.Tensor([Encode[cls]])
        rebb= torch.Tensor(rebb)
        #print(img.shape, cls.shape, rebb.shape)
        #print( cls, rebb)
        return img, cls, rebb



if __name__ == '__main__':
    pass
