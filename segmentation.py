import numpy as np
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv
from skimage.transform import resize

def segmentation_model(img, path):

    model = xrv.baseline_models.chestx_det.PSPNet()

    # img = skimage.io.imread("/content/MIMIC-CXR.jpeg")
    img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
    img = resize(img, (507, 507, 1))

    # if len(img.shape) == 2:
    #     img = img[:, :, 0]

    img = img.mean(2)[None, ...]  # Make single color channel

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])

    img = transform(img)
    img = torch.from_numpy(img)
    with torch.no_grad():
        pred = model(img)

    plt.figure(figsize=(26, 5))
    plt.subplot(1, len(model.targets) + 1, 1)
    plt.imshow(img[0], cmap='gray')
    for i in range(len(model.targets[2:6])):
        plt.subplot(1, len(model.targets) + 1, i + 2)
        plt.imshow(pred[0, i])
        plt.title(model.targets[i])
        plt.axis('off')


    plt.tight_layout()

    plt.savefig(path,  bbox_inches='tight', dpi = 100)
    plt.savefig("static/segmentation.jpg",  bbox_inches='tight', dpi = 100)