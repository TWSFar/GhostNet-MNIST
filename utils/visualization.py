import matplotlib.pyplot as plt


def show_image(image, label):
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1).imshow(image)
    plt.subplot(1, 2, 2).text(0.5, 0.5, str(label), fontsize=20, ha='center')
    # plt.subplot(1, 2, 2).set_axis_off()
    plt.show()
