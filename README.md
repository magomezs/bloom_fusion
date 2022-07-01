# bloom_fusion

This repository contains 4 files, to perform 3 diffrente tasks. All of them are meant to perform data augmentation of databases with images of reservoirs and rivers. The main goal is to insert blooms over these images.


# Apply style from one image to the other
- apply_style.py. This code uses a pretrained style transfer model.

# Crooping bloom regions
- crop_style_images.py. This code presents a semi-automatic application to extract and save interesting bloom areas in wide-range images.

# Bloom Fusion
- bloom_insertation.py. This is the main file. This presents an semi-automatic app to insert and fusión a bloom image with a wider-range image with reservois or rivers.
This allows to use both real bloom images or synthetic images
- im_processing.py. This file contains the functions requiered to perform the fusion.
