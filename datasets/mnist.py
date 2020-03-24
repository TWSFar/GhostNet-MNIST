import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import datasets, transforms


class RandomGaussianBlur(object):
    def __call__(self, image):
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return image


class RandomSaltPepperBlur(object):
    def __call__(self, image, prob=0.05):
        image = np.array(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    image[i][j] = round(random.random()) * 255

        return Image.fromarray(image)


def Mnist(data_dir="data", input_size=(224, 224), train=True):
    if train:
        tsf = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(input_size),
            RandomGaussianBlur(),
            RandomSaltPepperBlur(),
            transforms.ToTensor(),
        ])
    else:
        tsf = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
    dataset = datasets.MNIST(data_dir,
                             train=train,
                             transform=tsf,
                             download=True)

    return dataset


if __name__ == '__main__':
    dataset = Mnist()
    from torch.utils.data import DataLoader
    mm = DataLoader(dataset)
    for i, sample in enumerate(mm):
        max = sample[0].max()
        plt.imshow(sample[0].numpy().reshape((224, 224)))
        plt.show()
