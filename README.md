# Cycle-CDM

This is the pytorch implementation of our manuscript "Unsupervised Diffusion MRI Artefact Correction via Cycle Conditional Diffusion Model".

**Abstract**

Diffusion magnetic resonance imaging (MRI) is a widely employed technique for investigating the microstructural characteristics of biological tissues. However, the diffusion MRI data obtained is contaminated by artefacts (i.e., noise, etc.), which undermine subsequent analyses and interpretations. Artefact correction of diffusion MRI is therefore essential for improving its quality and reliability. To address this issue, we propose an unsupervised method (Cycle-CDM) for diffusion MRI artefact correction, which is based on a cycle conditional diffusion model architecture. In contrast to supervised methods that rely on paired data for training, Cycle-CDM overcomes the limitations of lacking ground truth references by utilizing two diffusion models and incorporating cycle consistency. By formulating artefact correction as a problem of conditional diffusion modeling, we are able to effectively learn the underlying noise distribution and successfully remove artefacts from diffusion MRI data. Experimental results on Growing Up in Singapore Towards Healthy Outcomes (GUSTO) datasets demonstrate that Cycle-CDM outperforms state-of-the-art methods in terms of artefact correction performance and the corrected data exhibits significantly improved medical applicability and accurately preserves the underlying tissue microstructure.

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
