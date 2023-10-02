# Cycle-CDM

This is the pytorch implementation of our manuscript "Unsupervised Diffusion MRI Artefact Correction via Cycle Conditional Diffusion Model".

**Abstract**

Diffusion-weighted magnetic resonance imaging (DWI) is a widely employed technique for investigating the microstructure and structural connectivity of the brain. However, the DWI data obtained is contaminated by artefacts (i.e., noise, motion, etc.), which undermine subsequent analyses and interpretations. To address this issue, we propose an unsupervised method (Cycle-CDM) for DWI artefact correction to improve its quality and reliability. Our method Cycle-CDM employs the cycle translation architecture to enables the generation of cycles between artefact-corrupted and artefact-free domains, ensuring the production of high-quality samples without paired data. On the basis, two conditional diffusion models were utilized to establishes data interrelation between domains, which utilizes translated fake images from cycle translation architecture as conditions to assist in generating the target image from the original noise. Furthermore, we design multiple specific constraints for Cycle-CDM to preserves accurate anatomical information of artefact-corrected DWI. By formulating artefact correction as a problem of conditional diffusion modeling, we are able to effectively learn the underlying noise distribution and successfully remove artefacts from DWI data. Experimental results on DWI datasets of children demonstrate that Cycle-CDM outperforms state-of-the-art methods (e.g., U-Net, CycleGAN, Pix2Pix and MUNIT) in terms of artefact correction performance and the corrected data exhibit accurately preserves the underlying tissue microstructure and hence significantly improved medical applicability.

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
