# Deep-learning-powered data analysis in plankton ecology

By Harshith Bachimanchi, Matthew I. M. Pinder, Chloé Robert, Pierre De Wit, Jonathan
Havenhand, Alexandra Kinnby, Daniel Midtvedt, Erik Selander and Giovanni Volpe.

The repository contains source code and data for the article, [Deep-learning-powered data analysis in plankton ecology](https://arxiv.org/).

<p align="center">
  <img width="600" src=https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/blob/1cb7a3147b2a1694e1e54373892fcbe3ffaa5134/assets/display_fig.png>
</p>

## Description

We provide examples of how to use deep learning for plankton data analysis. The examples are provided as Jupyter notebooks and can be run on Google Colab. The examples are divided into three categories:

- `detection-tutorials`: For plankton detection from microscopy videos
- `segmentation-tutorials`: For plankton segmentation and classification from microscopy images
- `trajectory-tutorials`: For plankton trajectory linking from microscopy videos

## Usage

### Detection-tutorials

We provide three examples of plankton detection from microscopy videos using deep learning. The examples are provided as Jupyter notebooks and can be found in [detection-tutorials](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/tree/main/detection-tutorials) folder. The pdf versions of the notebooks are included for quick preview. Alternatively, the notebooks can be run on Google Colab by clicking on the links below:

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/detection-tutorials/1-detection_plankton1.ipynb) [1-detection_plankton1.ipynb](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/detection-tutorials/1-detection_plankton1.ipynb) demonstrates the detection of plankton species _Oxyrrhis marina_ from microscopy videos.

2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/detection-tutorials/2-detection_plankton2.ipynb) [2-detection_plankton2.ipynb](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/detection-tutorials/2-detection_plankton2.ipynb) demonstrates the detection of plankton species _Dunaliella tertiolecta_ from microscopy videos.

3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/detection-tutorials/3-detection_plankton3.ipynb) [3-detection_plankton3.ipynb](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/detection-tutorials/3-detection_plankton3.ipynb) demonstrates the detection of plankton species _Isochrysis galbana_ from microscopy videos.

### Segmentation-tutorials

In these set of tutorials, we will see how to generate simulated datasets to train a deep learning model for plankton segmentation and classification. The examples are provided as Jupyter notebooks and can be found in [segmentation-tutorials](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/tree/main/segmentation-tutorials). The pdf versions of the notebooks are included for quick preview. Alternatively, the notebooks can be run on Google Colab by clicking on the links below:

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/segmentation-tutorials/4-simulating_planktons.ipynb) [4-simulation_plankton.ipynb](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/segmentation-tutorials/4-simulation_plankton.ipynb) demonstrates how to simulate images that look closer to experimental images using 'DeepTrack' software package. In this tutorial, we simulate images of plankton species _Noctiluca scintillans_ and _Dunaliella tertiolecta_ and use the simulated images to train a deep learning model for segmentation and classification. The training code is provided in the next tutorial notebook.

2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/segmentation-tutorials/5-training_UNet_segmentation.ipynb) [5-training_UNet_segmentation.ipynb](https://github.com/softmatterlab/Deep-learning-in-plankton-ecology/blob/main/segmentation-tutorials/5-training_UNet_segmentation.ipynb) demonstrates how to train a U-Net model for segmentation and classification of plankton species _Noctiluca scintillans_ and _Dunaliella tertiolecta_ using the simulated images generated in the previous tutorial. The trained model is then used to segment and classify the experimental images of the same species.

## Citation

If you use this code for your research, please consider citing our papers:

1. <a href="https://www.biorxiv.org" target="_blank">Deep-learning-powered data analysis in plankton ecology.</a>

```
Harshith Bachimanchi, Matthew I. M. Pinder, Chloé Robert, Pierre De Wit, Jonathan Havenhand, Alexandra Kinnby, Daniel Midtvedt, Erik Selander and Giovanni Volpe.
"Deep-learning-powered data analysis in plankton ecology."
bioRxiv (2021).
https://doi.org/
```

2. <a href="https://elifesciences.org/articles/79760" target="_blank">Microplankton life histories revealed by holographic microscopy and deep learning.</a>

```
Harshith Bachimanchi, Benjamin Midtvedt, Daniel Midtvedt, Erik Selander, Giovanni Volpe (2022).
"Microplankton life histories revealed by holographic microscopy and deep learning."
eLife 11:e79760.
https://doi.org/10.7554/eLife.79760
```

3. <a href="https://www.nature.com/articles/s41467-022-35004-y" target="_blank">Single-shot self-supervised object detection in microscopy.</a>

```
Midtvedt, B., Pineda, J., Skärberg, F. et al.
"Single-shot self-supervised object detection in microscopy."
Nat Commun 13, 7492 (2022).
```

4. <a href="https://www.nature.com/articles/s42256-022-00595-0" target="_blank">Geometric deep learning reveals the spatiotemporal features of microscopic motion.</a>

```
Jesús Pineda, Benjamin Midtvedt, Harshith Bachimanchi, Sergio Noé, Daniel  Midtvedt, Giovanni Volpe,1 and  Carlo  Manzo
"Geometric deep learning reveals the spatiotemporal fingerprint ofmicroscopic motion."
arXiv 2202.06355 (2022).
```

5. [DeepTrack2.1](https://github.com/softmatterlab/DeepTrack-2.0/tree/master):

```
Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt, Giovanni Volpe.
"Quantitative Digital Microscopy with Deep Learning."
Applied Physics Reviews 8 (2021), 011310.
https://doi.org/10.1063/5.0034891
```
