# Cycle-CDM

**Dependencies**

--python  3.8

--pytorch 1.13.0

--torchvision 0.14.0

--Pillow 9.4.0

--nibabel 5.2.1

--h5py 3.11.0

--numpy 1.23.5

--scipy 1.10.1

**Usages**

**Training**

First put the unpaired training data (noise and noise-free) into the “data” folder. Please run:

        python train.py

You will get the trained denoising weights in the “checkpoint” folder.


**Some available options:**

--data: Training and testing dataset.

--configs: Training and testing configuration parameters.

--checkpoint: Model training weights folder.

--diffusion: The necessary functions required for training and inference.


**Contact**

If you have any questions, please contact us (dlmu.p.l.zhu@gmail.com)

