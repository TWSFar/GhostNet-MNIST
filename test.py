import os
import cv2
import torch
import argparse
import numpy as np
import os.path as osp
from models.model import Model
from utils import show_image
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--image', type=str, default='/home/twsf/work/GhostNet-MNIST/examples/r2.png',
                        help='image path or directory(type in [jpg, png, JPEG])')
    parser.add_argument('--checkpoint', type=str, default='/home/twsf/work/GhostNet-MNIST/work_dir/model_best.pth')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    return args


# Test
def test():
    args = parse_args()
    print(args)
    if osp.isdir(args.image):
        img_list = os.listdir(args.image)
        images = [osp.join(args.image, img_name) for img_name in img_list
                  if osp.splitext(img_name)[1] in ['.jpg', '.png', '.JPEG']] 
    elif osp.isfile(args.image):
        images = [args.image]
    else:
        raise FileExistsError
    model = Model(checkpoint=args.checkpoint).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for image in images:
            img = cv2.imread(image, cv2.COLOR_BGR2GRAY)
            tsf_img = np.resize(img, (224, 224))
            input = torch.tensor(tsf_img, dtype=torch.float).view(1, 1, 224, 224)
            output = model(input.to(DEVICE))
            predict = torch.argmax(output)
            print(predict.item()+1)
            if args.show:
                show_image(img, predict.item()+1)


if __name__ == '__main__':
    test()
