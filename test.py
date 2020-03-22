import os
import cv2
import torch
import argparse
import os.path as osp
from models.model import Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--image', type=str, default='',
                        help='image path or directory(type in [jpg, PNG, JPEG])')
    parser.add_argument('--checkpoint', type=str, default='')
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
                  if osp.splitext(img_name)[1] in ['.jpg', '.PNG', '.JPEG']] 
    elif osp.isfile(args.image):
        images = [args.image]
    else:
        raise FileExistsError
    model = Model(checkpoint=args.checkpoint).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for image in images:
            img = cv2.imread(image)
            img = torch.tensor(img.resize((224, 224))).view(1, 1, 224, 224)
            output = model(img.to(DEVICE))
            predict = torch.argmax(output)
            print(predict)
            if args.show:
                pass


if __name__ == '__main__':
    test()
