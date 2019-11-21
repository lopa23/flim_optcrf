# flim_optcrf • [![Build Status][travis-image]][travis] [![PyPi][pypi-image]][pypi] [![License][license-image]][license]

[travis-image]: https://travis-ci.org/locuslab/qpth.png?branch=master
[travis]: http://travis-ci.org/locuslab/qpth

[pypi-image]: https://img.shields.io/pypi/v/qpth.svg
[pypi]: https://pypi.python.org/pypi/qpth

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

*A software for learning from FLIM data, using Optnet and CRF 


Data Setup
---------

Create a train and test directory. Setup the path of this in test1.py
Under each, there will be subdirectory for each image, say im3. There will be 4 files that need to be in each subdirectory
00ng_im3_NADH_PC102_60s_40x_oil_intensity_image.bmp #intensity image
00ng_im3_NADH_PC102_60s_40x_oil_truth.bmp #GT segmentation, created by matlab code
H.mat #parmeters for running the QP, created by matlab code
label.txt #annotation, created by matlab code
---


How to run Matlab code
----------------------

1. Go to folder matlab_code
2. Replace all occurence of imA(where A is a number) by the image you want to process in read_asc_new.m and GenerateQP_params.m
3. Run read_asc_new


Run Python Code
-------------

The main function to call here is test1.py: This can be called as python test1.py
The code here utilizes two packages:
QPTH: for solving QPs
----
Downloaded from https://github.com/locuslab/qpth/
This code was modified to work with our code. Please install all dependencies needed to run this including Python, Torch, Cuda, Numpy(see link above)


Permutohedral Filtering
-----------------------
This code also utilizes the Pytorch implementation of CRFasRNN
https://www.robots.ox.ac.uk › ~szheng › papers › CRFasRNN
https://github.com › sadeepj › crfasrnn_keras
This has been adapted from
https://github.com/Fettpet/pytorch-crfasrnn
Download and install the Permutohedral_Filtering from here