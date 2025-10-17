"""
01. Segment anything
====================

The Segment Anything Model (SAM) produces high quality object masks from input
prompts such as points or boxes, and it can be used to generate masks for all
objects in an image. It has been trained on a dataset of 11 million images and
1.1 billion masks, and has strong zero-shot performance on a variety of
segmentation tasks.

.. raw:: html

   <center>

      <a href="https://pypi.org/project/segment-anything-py/" target="_blank">
         <button class="btn-github"> PyPi
            <img class="btn-icon" src="https://pypi.org/static/images/logo-small.2a411bc6.svg" width=18/>
         </button>
      </a>

      <a href="https://segment-anything.com/demo" target="_blank">
         <button class="btn-github"> Demo
            <img class="btn-icon" src="./../../_static/images/icon-meta.png" width=18/>
         </button>
      </a>

      <a href="https://github.com/facebookresearch/segment-anything" target="_blank">
         <button class="btn-github"> Github
            <img class="btn-icon" src="./../../_static/images/icon-github.svg" width=18/>
         </button>
      </a>

      <a href="https://github.com/facebookresearch/segment-anything/tree/main/notebooks" target="_blank">
         <button class="btn-github"> Notebooks
            <img class="btn-icon" src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" width=18/>
         </button>
      </a>

   </center>

    <br>
    <br>

.. note:: It is required to download the checkpoints first!

"""
# https://pypi.org/project/segment-anything-py/

# Libraries
import numpy as np
import torch
import matplotlib.pyplot as plt

# .. note: The notebook uses cv2 and does some alteration to the image.
# import cv2

# Library
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry


def show_anns(anns, ax=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

# Constant
CHECKPOINTS = {
    'vit_b': './objects/main01/sam_vit_b_01ec64.pth', # 0.37 GB
    'vit_l': './objects/main01/sam_vit_l_0b3195.pth', # 1.2 GB
    'vit_h': './objects/main01/sam_vit_h_4b8939.pth', # 2.4 GB
}

# Variables
model = 'vit_b'

# Load image
image = plt.imread('./objects/main01/photo-1.jpg')

# Load model
sam = sam_model_registry[model](checkpoint=CHECKPOINTS[model])

# Create mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Show
print("Computing masks....")

# Compute masks
masks = mask_generator.generate(image)

# Display
_, axs = plt.subplots(1, 2, figsize=(20,20), sharey=True)
axs[0].imshow(image)
axs[1].imshow(image)
axs[0].axis('off')
axs[1].axis('off')
show_anns(masks, ax=axs[1])

plt.tight_layout()
plt.show()