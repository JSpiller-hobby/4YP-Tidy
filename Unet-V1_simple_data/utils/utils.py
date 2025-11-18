#original: https://github.com/milesial/Pytorch-UNet
#Licensed under GNU GENERAL PUBLIC LICENSE V3
#MODIFICATION: now plots vector fields instead of mask; added plotting for mask only;

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Plots an input PIL image and vector field (mask)
def plot_img_and_mask(img, mask, mask_title):
    
    img.show()
    width = img.size[0]
    height = img.size[1]
    X, Y = np.meshgrid(range(width), range(height))
    U = mask[0,:,:]
    V = mask[1,:,:]
    plt.quiver(X, Y, U, V, color='g', scale_units = 'xy')
    plt.title(mask_title)
    plt.grid()
    plt.gca().set_aspect("equal")
    #plt.show()

def plot_mask(mask, mask_title):
    width = 256
    height = 256
    X, Y = np.meshgrid(range(width), range(height))
    U = mask[:,:,0]
    V = mask[:,:,1]
    plt.quiver(X, Y, U, V, color='g', scale_units = 'xy')
    plt.title(mask_title)
    plt.grid()
    plt.gca().set_aspect("equal")
    #plt.show()