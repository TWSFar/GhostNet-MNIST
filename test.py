import os
import cv2
import torch
import argparse
import os.path as osp
from PIL import Image
from utils import show_image
from models.model import Model
from torchvision import transforms
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--image', type=str, default='/home/twsf/work/GhostNet-MNIST/demo',
                        help='image path or directory(type in [jpg, png, JPEG])')
    parser.add_argument('--checkpoint', type=str, default='/home/twsf/work/GhostNet-MNIST/work_dir/last.pth')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    return args


def test():
    # Configs
    args = parse_args()
    print(args)

    # Test images and transforms
    if osp.isdir(args.image):
        img_list = os.listdir(args.image)
        images = [osp.join(args.image, img_name) for img_name in img_list
                  if osp.splitext(img_name)[1] in ['.jpg', '.png', '.JPEG']] 
    elif osp.isfile(args.image):
        images = [args.image]
    else:
        raise FileExistsError
    tsf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    # Model
    model = Model(checkpoint=args.checkpoint).to(DEVICE)

    # Test
    model.eval()
    with torch.no_grad():
        for image in images:
            # img = Image.open(image).convert('L')
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            img = Image.fromarray(img)
            input = tsf(img).unsqueeze(0)
            output = model(input.to(DEVICE))
            predict = torch.argmax(output)
            print('image: {}'.format(image))
            print('result: {}'.format(predict.item()))

            # Show image and result
            if args.show:
                show_image(img, predict.item())


if __name__ == '__main__':
    test()
