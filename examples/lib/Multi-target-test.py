import os
import cv2
import torch
import argparse
import numpy as np
import os.path as osp
from models.model import Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--image', type=str, default='/home/twsf/work/GhostNet-MNIST/examples/r2.png',
                        help='image path(type in [jpg, png, JPEG])')
    parser.add_argument('--checkpoint', type=str, default='/home/twsf/work/GhostNet-MNIST/work_dir/model_best.pth')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    return args


# Test
def test():
    args = parse_args()
    print(args)
    if not osp.isfile(args.image):
        raise FileExistsError
    img = np.resize(cv2.imread(args.image), (224, 224))
    model = Model(checkpoint=args.checkpoint).to(DEVICE)
    model.eval()
    with torch.no_grad():
        input = torch.tensor(img, dtype=torch.float).view(1, 1, 224, 224)
        output = model(input.to(DEVICE))
        predict = torch.argmax(output)
        print(predict.item()+1)
        if args.show:
            pass


if __name__ == '__main__':
    test()
