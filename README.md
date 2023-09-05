# Cycle-CDM

This is the pytorch implementation of our manuscript "Unsupervised Diffusion MRI Artefact Correction via Cycle Conditional Diffusion Model".

**Abstract**

Diffusion magnetic resonance imaging (MRI) is a widely employed technique for investigating the microstructural characteristics of biological tissues. However, the diffusion MRI data obtained is contaminated by artefacts (i.e., noise, etc.), which undermine subsequent analyses and interpretations. Artefact correction of diffusion MRI is therefore essential for improving its quality and reliability. To address this issue, we propose an unsupervised method (Cycle-CDM) for diffusion MRI artefact correction, which is based on a cycle conditional diffusion model architecture. In contrast to supervised methods that rely on paired data for training, Cycle-CDM overcomes the limitations of lacking ground truth references by utilizing two diffusion models and incorporating cycle consistency. By formulating artefact correction as a problem of conditional diffusion modeling, we are able to effectively learn the underlying noise distribution and successfully remove artefacts from diffusion MRI data. Experimental results on Growing Up in Singapore Towards Healthy Outcomes (GUSTO) datasets demonstrate that Cycle-CDM outperforms state-of-the-art methods in terms of artefact correction performance and the corrected data exhibits significantly improved medical applicability and accurately preserves the underlying tissue microstructure.

**Dependencies**

--python  3.6

--pytorch 1.7.0

--torchvision 0.8.0

--scipy 1.5.4

--pliiow 8.4.0

--timm 0.4.9

--openpyxl 3.0.7

**Usages**

**Testing batch images**

To run the test_batch, please put trained model in 'model' folder, then run:

        python test_batch.py

You will get a quality score and a distortion type, and a higher score indicates better image quality.

**Training & Testing on IQA Databases**

First put the pretrained weight (https://drive.google.com/file/d/1W0mvaqjFVlEXSZynEJhRRg7LlsCr2AV9/view?usp=sharing) in 'pretrained' folder. Training and testing our model on the specified dataset.

        python train_test_IQA.py

**Some available options:**

--dataset: Training and testing dataset, support datasets: livec | koniq-10k | bid | live | csiq.

--train_patch_num: Sampled image patch number per training image.

--test_patch_num: Sampled image patch number per testing image.

--batch_size: Batch size.

**Acknowledgement**

The code used in this research is inspired by HyperIQA (https://github.com/SSL92/hyperIQA) and ResT (https://github.com/wofmanaf/ResT).

**Contact**

If you have any questions, please contact us (dlmu.p.l.zhu@gmail.com)
