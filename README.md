# Implemeting_data_preparation_of_RCNN_paper
 implementing data preparation of RCNN paper ![link](https://arxiv.org/abs/1311.2524) from scratch as follow:
 1) download VOC dataset .tar file
 2) extract dataset file
 3) for each image :
 4) - parse its XML file to extract bounding box (bb) and corresponding class
 5) - perform the selective search to get the bb proposals
 6) - for each bb resulted from step (5):
 7) - - perform NMS with IoU of .5 with groundtruth else assign background label for it 
 8) - - calculate the groundtruth bb offset (used for bb regression ) 
 9) return computed bb with its assigned label from step (7)
