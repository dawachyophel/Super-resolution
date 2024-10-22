This package contains the MATLAB code which is associated with the following paper:

Lepcha, Dawa Chyophel, Bhawna Goyal, Ayush Dogra, and Shui‐Hua Wang. "An efficient medical image super resolution based on piecewise linear regression strategy using domain transform filtering." Concurrency and Computation: Practice and Experience 34, no. 20 (2022): e6644. (https://doi.org/10.1002/cpe.6644)

Building the code:
There is only a single mex/cpp file included in this package that can be build with the MATLAB function "mex". The steps are:
mex -setup C++                                                                 
mex SR_scale_Hadamard.cpp ("scale" is the upscaling factor,e.g. mex SR_2_Hadamard.cpp )
We provide a pre-compiled file "SR_2_Hadamard.mexw64" for Windows (64 bit).

Using the code:
The file "DC_demo.m" is the demo usage of the proposed algorithm. However the file contains one test image (MRI) c03_1.bmp. The remaining test images can be download from 
http://splab.cz/en/download/databaze/ultrasound (Ultrasound dataset)
https://data.mendeley.com/datasets/p9bpx9ctcv/2 (Angiography dataset)
https://www.med.harvard.edu/aanlib/home.html ( CT & MRI dataset)
https://dl.acm.org/doi/10.1145/3083187.3083212 (Endoscopic dataset)
https://stanfordmlgroup.github.io/competitions/mura/ (X-ray dataset)

The training data can be download from above dataset links.The traing data should be kept in a folder "Source1" in directory. The training dataset used in our paper can be generated by file "create_train_data.m". 

The modcrop function is from the source code of: C. Dong, C. C. Loy, K. He, and X. Tang. Learning a deep convolutional network for image super-resolution. In European Conference on Computer Vision, pages 184¨C199. Springer, 2014.

The learned mapping models as well as the decision tree for scaling factor 2 can be directly used for SR reconstruction. 

For demo purpose, we have presented results for scale factor 2 in given code. The results for scale factor 4 can be obtained by going through above given instructions. The quantitative results are compared using psnr & ssim for this study. 

Please refer to our paper for algorithm details.

Please refer to the above publication if you use this code. Thank you.

Do not hesitate to contact me if you meet any problems when using this code.
