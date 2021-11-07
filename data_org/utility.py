import xml.etree.ElementTree as ET
from data_org import *
import numpy as np
import cv2 as cv
import os


Decode = ['background', 'aeroplane', 'bicycle',
          'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
          'chair', 'cow', 'diningtable', 'dog',
          'horse', 'motorbike', 'person',
          'pottedplant', 'sheep', 'sofa',
          'train', 'tvmonitor', ]  # reverse Label Encoing
Encode = {x: i for i, x in enumerate(Decode)}  # label Encoding


# read images ids in the textfiles
def get_ids(labels_path):
    labels = list()
    with open(labels_path) as file:
        for line in file.readlines():
            labels.append(line[:-1])
    return labels


# parse the xml annoatation files to dictionary
def get_cls_bb(image_id):
    xmlfile_path = os.path.join(ANNOTAT_PATH, image_id + '.xml')
    root = ET.parse(xmlfile_path).getroot()
    image_data = dict()
    image_data['name'] = root.find('filename').text
    image_data['dims'] = (int(root.find('size/depth').text),
                          int(root.find('size/width').text),
                          int(root.find('size/height').text),)
    image_data['object'] = list()
    for obj in root.findall('object'):
        if int(obj.find('difficult').text):
            continue
        image_data['object'].append([
            obj.find('name').text,
            int(obj.find('bndbox/xmin').text),
            int(obj.find('bndbox/ymin').text),
            int(obj.find('bndbox/xmax').text),
            int(obj.find('bndbox/ymax').text),
        ])
    return image_data


# read aparticular image file with asecific id
def get_image(image_id):
    image_file_path = os.path.join(IMAGE_PATH, image_id + '.jpg')
    img_file = cv.imread(image_file_path)
    return img_file


# finds the intersection box between two bb
# does not care about spicaial cases
def get_inter_bb(bb1, bb2):
    xmin1, ymin1, xmax1, ymax1 = bb1
    xmin2, ymin2, xmax2, ymax2 = bb2
    bb = (max(xmin1, xmin2), max(ymin1, ymin2),
          min(xmax1, xmax2), min(ymax1, ymax2))
    return bb


# calculate the area of particular bb and handle the mis cased of get_inter_bb
def get_bb_area(bb):
    xmin1, ymin1, xmax1, ymax1 = bb
    return max((xmax1 - xmin1), 0) * max((ymax1 - ymin1), 0)


# calculate the Intersection over union of two bb
def IoU(bb1, bb2):
    ibb = get_inter_bb(bb1, bb2)
    x = get_bb_area
    return x(ibb) / (x(bb1) + x(bb2) - x(ibb))


def crop_image(im, bb):
    x1, y1, x2, y2 = bb
    cr = im[y1:y2, x1:x2]
    if cr.shape[0] and cr.shape[1]:
        return cr


# return as bb resulted from the selective search
def selective_Search(img):
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rects = rects.tolist()
    out = list()
    for xm, ym, w, h in rects:
        out.append([xm, ym, xm + w, ym + h])
    return out


# separate the bb reulted from selective search to positive or negative
# crops based on IoU score must be higher than IoU_tr
# and return a list of positive and negative crops
def selective_search_out_cleaning(rects, gtbb,
                        IoU_tr=.5, area_tr=.5):
    posit_candidbb = gtbb[:]
    negat_candidbb = list()
    min_area = 2000
    for gbb in gtbb:
        if get_bb_area(gbb[1:]) < min_area:
            min_area = get_bb_area(gbb[1:])
    min_area *= area_tr
    for bb in rects:
        is_back = True
        for gbb in gtbb:
            if IoU(bb, gbb[1:]) > IoU_tr:
                posit_candidbb.append(gbb[0:1] + bb)
                is_back = False
        if is_back and get_bb_area(bb) > min_area:
            negat_candidbb.append(['background'] + bb)
    return posit_candidbb, negat_candidbb

#assiging the selective search out to the most overlapped gtbb
def selective_out_clustering(rects, gtbb,
                        IoU_tr=.5, area_tr=.5):
    rects = np.array(rects)
    scores = np.zeros( (rects.shape[0],))
    rects_labels = np.zeros( (rects.shape[0],),dtype =int)
    reff_gtbb = np.zeros_like(rects)

    for i in range( rects.shape[0]) :
        for gbb in gtbb :
            iou = IoU(rects[i].tolist(), gbb[1:])
            if scores[i] < iou and IoU_tr < iou:
                rects_labels[i] = Encode[gbb[0]]
                reff_gtbb[i] = gbb[1:]

    reff_gtbb = reff_gtbb - rects ########################
    nega = list()
    posi = gtbb[:]
    reff_po = [[0,0,0,0] for _ in range( gtbb.__len__())]
    reff_nega = list()
    for i in range( rects_labels.shape[0] ):
        if rects_labels[i]:
            posi.append([Decode[rects_labels[i]]] + rects[i].tolist())
            reff_po.append( reff_gtbb[i].tolist() )
        else:
            nega.append([Decode[rects_labels[i]]] + rects[i].tolist())
            reff_nega.append(reff_gtbb[i].tolist())
    return posi, nega, reff_po ,reff_nega


