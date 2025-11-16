#original: https://github.com/milesial/Pytorch-UNet
#Licensed under GNU GENERAL PUBLIC LICENSE V3
#MODIFICATION: removed code to handle carvana data; removed code which extracted unique mask values and used them to normalise the mask; changed getitem() to use the mask as a np array, not image;
# removed mask_suffix parameter; removed scanning of masks for unique classes; removed treatment of data with only one channel; mask values returned as float 32; 

#TODO: implement rescaling; currently broken because it is not a quick fix to rescale the velocity fields (numpy arrays)

import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def load_image(filename):
    """use PIL to load an image from numpy array, pytorch tensor, or any standard image format """
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'                  #NOTE: SCALE NOT IMPLEMENTED
        self.scale = scale

        #Get ids, a list of the image file names (str), without extensions.
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')] 
        if not self.ids:                                                                                                                                                                                                                                       
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')                                                                                                                                                     

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(datum, scale, is_mask):
        #----------------------------------------------------------------------------------------------NOT YET IMPLEMENTED SCALING, AS IT IS NON-TRIVIAL FOR VELOCITY FIELD
        #rescale, convert to np array
        # w = np.asarray(datum).shape[1]
        # h = np.asarray(datum).shape[0]
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #datum = datum.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC) 
        #------------------------------------------------------------------------------------------------
        img = np.asarray(datum)
        
        img = img.transpose((2, 0, 1)) 

        #for images, set all values between 0 and 1.
        if not is_mask:
            if (img > 1).any():
                img = img / 255.0

        return img

    #Use ids to search for image-mask pairs (by file name), for a given idx in ids.
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = np.load(mask_file[0])
        img = load_image(img_file[0])

        assert np.asarray(img).shape[0] == mask.shape[0] and np.asarray(img).shape[1] == mask.shape[1], \
            f'Image and mask {name} should be the same size, but are {np.asarray(img).shape} and {mask.shape}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }
