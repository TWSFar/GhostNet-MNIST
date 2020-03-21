import matplotlib.pyplot as plt

def show_image(image, label):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(image[:, :, ::-1])
    plt.show()
    pass