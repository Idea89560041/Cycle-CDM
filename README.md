# Cycle-CDM

This is the pytorch implementation of our manuscript "Unsupervised Diffusion MRI Artefact Correction via Cycle Conditional Diffusion Model".


**Dependencies**

--python  3.8.16

--pytorch 1.12.1

--torchvision 0.13.1

--matplotlib 3.5.2

--numpy 1.24.3

--einops 0.6.0

--tqdm 4.64.0

**Usages**

**Testing artefact correction demo**

To run the demo, please put trained model (https://drive.google.com/file/d/1vNodDdD7Rzd7YmvKHWYbcH5ihQJL1R6O/view?usp=sharing) in 'checkpoint' folder, then run:

        python ./diffusion/sampling.py

You will get the artefact-free image in /data/artefact-free.

**Acknowledgement**

The code used in this research is inspired by UNIT_DDPM (https://github.com/konkuad/UNIT-DDPM-Unofficial) and F-LSeSim (https://github.com/lyndonzheng/F-LSeSim).

**Contact**

If you have any questions, please contact us (dlmu.p.l.zhu@gmail.com)
