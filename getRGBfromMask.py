#!/usr/local/bin/python3
from PIL import Image
import numpy as np
import pdb
# Open the input image as numpy array
npImage=np.array(Image.open("original.png"))
# Open the mask image as numpy array
npMask=np.array(Image.open("mask.png").convert("RGB"))
#pdb.set_trace()
# Make a binary array identifying where the mask is black
cond = npMask<128

# Select image or mask according to condition array
pixels=np.where(cond, npImage, npMask)

# Save resulting image
result=Image.fromarray(pixels)
result.save('result.png')
