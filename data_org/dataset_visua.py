from data_org import *
import cv2 as cv


# pass a sequence of images ids and will be displayed with the cooresponding bb
def bb_imshow(image_ids,
              winame='image_sequence'):
    font = cv.FONT_HERSHEY_SIMPLEX
    for id in image_ids:
        image, mdt = get_image(id), get_cls_bb(id)
        cv.putText(image, id, (0, mdt['dims'][2] - 10), font,
                   fontScale=.5, color=(0, 255, 0),
                   thickness=1)
        image_with_BB(image, mdt['object'])
        cv.imshow(winame, image)
        if chr(cv.waitKey(0) & 0xff) == 'q':
            break
    cv.destroyAllWindows()


# given a set of alabeled bbs and an image it will adraw each bb over the image
def image_with_BB(im, bbs):
    font = cv.FONT_HERSHEY_SIMPLEX
    for cls, xmin, ymin, xmax, ymax in bbs:
        cv.rectangle(im,
                     (xmin, ymin), (xmax, ymax),
                     (255, 0, 0), 2)
        cv.putText(im, cls, (xmin + 2, ymin + 16), font,
                   fontScale=.5, color=(0, 255, 0),
                   thickness=1)


def imshow_posit_nega_bb(id):
    image, mdt = get_image(id), get_cls_bb(id)
    reacts = selective_Search(image)
    po, ne = selective_search_out_cleaning(reacts, mdt['object'][:])
    image_with_BB(image, po[0:1] + ne[0:1])
    cv.imshow('test', image)
    if chr(cv.waitKey(0) & 0xff) == 'q':
        pass
    cv.destroyAllWindows()


def show_the_resulted_dataset_after_crops():
    dts = RCNN_ready_dataset(TRAINVAL_LABELS)
    i = 0
    while (chr(cv.waitKey(0) & 0xff) != 'q'):
        im, lbl, reff = dts.__getitem__(i)
        print(lbl, i, reff, im.shape)
        im = im.permute(1,2,0).detach().numpy()

        cv.imshow("test", im)
        i += 1
    cv.destroyAllWindows()


if __name__ == '__main__':
    l = get_ids(TRAINVAL_LABELS)
    image_data = get_cls_bb(l[0])
    img = get_image(l[0])
    x = 500
    bb_imshow(l[x:x + 10])
    # imshow_posit_nega_bb(l[320])
    show_the_resulted_dataset_after_crops()
    """
    cv.imshow('example', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
