# Awesome-multimodal-semantic-segmentation
Resources for multimodal semantic segmentation

# 🧠 Multimodal Semantic Segmentation: Datasets & Resources Overview

This repository provides a curated list of **datasets** and **literature** for multimodal semantic segmentation across **RGB-D**, **RGB-Thermal**, **RGB-Event**, **RGB-LiDAR**, **Audio-Visual**, **Medical**, and **Remote Sensing** domains.

---

## 📁 1. Multimodal Datasets by Modality

### 1.1 RGB + Depth (RGB-D)

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| NYU Depth V2 | Indoor Segmentation and Support Inference from RGBD Images | ECCV | 2012 | [Link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) |
| SUN RGB-D | SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite | CVPR | 2015 | [Link](https://rgbd.cs.princeton.edu/) |
| Stanford2D3D | 2D-3D-S: A Large-Scale Indoor Dataset for 3D Semantic Segmentation | CVPR | 2017 | [Link](http://buildingparser.stanford.edu/dataset.html) |
| ScanNetV2 | ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes | CVPR | 2017 | [Link](http://www.scan-net.org/) |
| Cityscapes (Stereo) | The Cityscapes Dataset for Semantic Urban Scene Understanding | CVPR | 2016 | [Link](https://www.cityscapes-dataset.com/) |
| SYNTHIA2Cityscapes&SELMA2Cityscapes | Source-Free Domain Adaptation for RGB-D Semantic Segmentation with Vision Transformers | WACV | 2023 | [Link](https://arxiv.org/abs/2305.14269) |
| NYUDv2 | Self-Enhanced Feature Fusion for RGB-D Semantic Segmentation | IEEE SPL | 2024 | [Link](https://ieeexplore.ieee.org/document/10706844/) |


---

### 1.2 RGB + Thermal (RGB-T)

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| KAIST Multispectral | Multispectral Pedestrian Detection: Benchmark Dataset and Baseline | CVPRW | 2015 | [Link](https://soonminhwang.github.io/rgbt-ped-detection/) |
| LLVIP | LLVIP: A Visible-Thermal Paired Dataset for Low-Light Vision | ECCV | 2022 | [Link](https://github.com/wyf0912/LLVIP) |
| PST900 | PST900: RGB-Thermal Dataset for Segmentation | IJCAI | 2021 | [Link](https://github.com/Vanint/PST900_RGBT) |
| MFNet | MS-IRTNet: Multistage information interaction network for RGB-T semantic segmentation | Information Sciences	 | 2021 | [Link](https://github.com/poisonzzw/MS-IRTNet) |
| MFNet | UTFNet: Uncertainty-Guided Trustworthy Fusion Network for RGB-Thermal Semantic Segmentation | IEEE | 2023 | [Link](https://github.com/KustTeamWQW/UTFNet) |
| MFNet&PST900 | Region-adaptive and context-complementary cross modulation for RGB-T semantic segmentation | Pattern Recognition | 2024 | [Link](https://linkinghub.elsevier.com/retrieve/pii/S0031320323007896) |
| MFNet&PST900 | RGB-T Semantic Segmentation With Location, Activation, and Sharpening | TCSVT | 2023 | [Link](https://github.com/MathLee/LASNet) |
| MVSeg | Multispectral Video Semantic Segmentation: A Benchmark Dataset and Baseline | CVPR | 2023 | [Link](https://ieeexplore.ieee.org/document/10203299/) |
| MFNet&PST900 | MMSMCNet: Modal Memory Sharing and Morphological Complementary Networks for RGB-T Urban Scene Semantic Segmentation | IEEE | 2023 | [Link](https://github.com/2021nihao/MMSMCNet) |
| MFNet&PST900 | Mitigating Modality Discrepancies for RGB-T Semantic Segmentation | IEEE | 2024 | [Link](https://ieeexplore.ieee.org/document/10008228/) |
| MFNet&PST900 | GMNet: Graded-Feature Multilabel-Learning Network for RGB-Thermal Urban Scene Semantic Segmentation |  | 2023 | [Link](https://ieeexplore.ieee.org/document/9531449/) |
| MFNet | FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time Semantic Segmentation | IEEE | 2021 | [Link](https://ieeexplore.ieee.org/document/9636084/) |
| MFNet&PST900 | Complementarity-aware cross-modal feature fusion network for RGB-T semantic segmentation | Pattern Recognition | 2022 | [Link](https://linkinghub.elsevier.com/retrieve/pii/S0031320322003624) |
| MFNet | ABMDRNet: Adaptive-weighted Bi-directional Modality Difference Reduction Network for RGB-T Semantic Segmentation | CVPR | 2021 | [Link](https://ieeexplore.ieee.org/document/9578077/) |

---

### 1.3 RGB + Event Camera

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| DSEC | The DSEC Dataset for Event-based Stereo Visual Odometry | RA-L + ICRA | 2021 | [Link](https://dsec.ifi.uzh.ch/) |
| ESS | ESS: Learning Event-based Semantic Segmentation from Still Images | CVPR | 2021 | [Link](https://github.com/lyuchenyang/ESS) |
| DDD17 | Driving Dataset for Event Cameras (DDD17) | arXiv | 2017 | [Link](https://github.com/uzh-rpg/rpg_davis_data) |
| DSEC | SAM-Event-Adapter: Adapting Segment Anything Model for Event-RGB Semantic Segmentation | ICRA | 2024 | [Link](https://ieeexplore.ieee.org/document/10611127/) |

---

### 1.4 RGB + LiDAR

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| SemanticKITTI | SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences | ICCV | 2019 | [Link](http://www.semantic-kitti.org/) |
| A2D2 | Audi A2D2: AEV Autonomous Driving Dataset | Dataset Release | 2020 | [Link](https://www.a2d2.audi/) |
| nuScenes | nuScenes: A Multimodal Dataset for Autonomous Driving | CVPR | 2020 | [Link](https://www.nuscenes.org/) |

---

### 1.5 RGB + Audio

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| AVSBench | Unveiling and Mitigating Bias in Audio-Visual Segmentation | ECCV | 2022 | [Link](https://github.com/OpenGVLab/AVSBench) |
| MUSIC-AVS | MUSIC: A Multimodal Dataset for Sound Source Localization | ECCV | 2018 | [Link](https://zenodo.org/record/3402610) |

---

### 1.6 RGB + N (Multiple Modalities)

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| ArbitraryModalSeg | Delivering Arbitrary-Modal Semantic Segmentation | CVPR | 2023 | [Link](https://arxiv.org/pdf/2303.01480) |
| Multimodal Material Segmentation | Segmenting Materials from Local Appearance and Global Context | CVPR | 2022 | [Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Multimodal_Material_Segmentation_CVPR_2022_paper.pdf) |

---

### 1.7 Medical Multimodal Datasets

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| BraTS | The Multimodal Brain Tumor Image Segmentation Benchmark | IEEE TMI | 2015 | [Link](https://doi.org/10.1109/TMI.2014.2377694) |
| AMOS | AMOS: Abdominal Multi-organ Benchmark | MICCAI | 2022 | [Link](https://amos22.grand-challenge.org/) |
| CHAOS | Combined Healthy Abdominal Organ Segmentation | arXiv | 2019 | [Link](https://arxiv.org/abs/1911.11320) |
| MM-WHS | Whole Heart Segmentation Benchmark | MedIA | 2019 | [Link](https://doi.org/10.1016/j.media.2019.01.012) |
| SegTHOR | Segmentation of Thoracic Organs at Risk | arXiv | 2019 | [Link](https://arxiv.org/abs/1902.09063) |

---

### 1.8 Remote Sensing Multimodal Datasets

| Dataset | Paper Title | Venue | Year | Link |
|---------|-------------|-------|------|------|
| ISPRS Vaihingen / Potsdam | ISPRS Urban Object Classification Benchmark | ISPRS Annals | 2012 | [Link](https://www2.isprs.org/commissions/comm2/wg4/benchmark/semantic-labeling/) |
| LoveDA | LoveDA: Remote Sensing Domain Adaptive Segmentation | NeurIPS (Datasets Track) | 2021 | [Link](https://github.com/Junjue-Wang/LoveDA) |
| DASE2021 | Towards Cross-Modality Domain Adaptation for RS | IGARSS | 2021 | [Link](https://github.com/Junjue-Wang/DASE2021) |
| Houston2018 | GRSS Data Fusion: Hyperspectral + LiDAR | JSTARS | 2018 | [Link](https://www.grss-ieee.org/community/technical-committees/data-fusion/) |

---

## 🔀 2. Method Design: Architectures, Training

### 2.1 Architecture Design

#### 2.1.1 Modality Interaction Design

| Paper Title | Venue | Year | Modality | Link |
|-------------|-------|------|----------|------|
| DFORMER: Rethinking RGBD Representation Learning | CVPR | 2022 | RGB-D | [arXiv](https://arxiv.org/abs/2111.15645) |
| StitchFusion: Weaving Any Visual Modalities | CVPR | 2023 | RGB+N | [arXiv](https://arxiv.org/abs/2304.14302) |

#### 2.1.2 Modality-Invariant Representation

| Paper Title | Venue | Year | Modality | Link |
|-------------|-------|------|----------|------|
| DMR: Decomposed Multi-Modality Representations | NeurIPS | 2022 | RGB + Event | [arXiv](https://arxiv.org/abs/2211.08410) |
| Multi-interactive Feature Learning Benchmark | IJCV | 2021 | RGB-T | [Springer](https://link.springer.com/article/10.1007/s11263-021-01474-9) |

#### 2.1.3 Prompt-Based Learning

| Paper Title | Venue | Year | Modality | Link |
|-------------|-------|------|----------|------|
| SDSTrack: Self-Distillation Symmetric Adapter | CVPR | 2023 | Multi-modal | [arXiv](https://arxiv.org/abs/2303.01977) |
| Visual Prompt for Multi-Modal Tracking | NeurIPS | 2022 | RGB-T | [arXiv](https://arxiv.org/abs/2210.10904) |
| X-Prompt: Cross-modal Prompt for VOS | CVPR | 2023 | Video + Prompt | [arXiv](https://arxiv.org/abs/2304.04223) |
| UniDSeg: Unified Prompt for 3D Segmentation | CVPR | 2024 | RGB + 3D | [arXiv](https://arxiv.org/abs/2311.00277) |
| Dual-Prompt Learning for Efficient Segmentation | ECCV | 2022 | RGB-T | [arXiv](https://arxiv.org/abs/2207.10983) |

---

### 2.2 Training Strategies

| Paper Title | Venue | Year | Method Type | Link |
|-------------|-------|------|-------------|------|
| Complementary Random Masking for RGB-T | ECCV | 2022 | Data Augmentation | [arXiv](https://arxiv.org/abs/2303.17386) |
| Rethinking Reverse Distillation for Multimodal Anomaly Detection | CVPR | 2023 | Knowledge Distillation | [arXiv](https://arxiv.org/abs/2303.02515) |
| Cross-Modal Contrastive Pretraining for Medical Fusion | MICCAI | 2023 | Self-supervised | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-43893-3_28) |
| Modality Dropout for Robust Fusion | TMI | 2022 | Modality Robustness | [IEEE](https://ieeexplore.ieee.org/document/9733185) |
| Joint Feature Regularization for Multi-Stream Fusion | TPAMI | 2023 | Feature Fusion | [IEEE](https://ieeexplore.ieee.org/document/10014115) |

---

 ---

## 🔀 3. Modality-Specific Fusion Techniques

### 3.1 RGB + Audio (Audio-Visual Semantic Segmentation)

| Paper Title | Venue | Year | Fusion Type | Link |
|-------------|-------|------|-------------|------|
| Can Textual Semantics Mitigate Sounding Object Segmentation Preference? | ECCV | 2022 | Audio-Guided Attention | [arXiv](https://arxiv.org/abs/2207.11230) |
| Unveiling and Mitigating Bias in Audio-Visual Segmentation | ECCV | 2022 | Audio-Visual Masking | [GitHub](https://github.com/OpenGVLab/AVSBench) |
| AudioScope: Spatial Audio Segmentation with Unsupervised Clustering | NeurIPS | 2023 | Cross-modal Fusion | [arXiv](https://arxiv.org/abs/2305.01521) |

#### 💡 Notes:
- Typical fusion involves **spectrogram encoding** of audio features (e.g., log-Mel), followed by late or attention-based integration into visual streams.
- Spatial alignment of audio and image cues is key in noisy environments or under occlusion.

---

### 3.2 RGB + Event (Event-based Semantic Segmentation)

| Paper Title | Venue | Year | Fusion Type | Link |
|-------------|-------|------|-------------|------|
| ESS: Learning Event-based Semantic Segmentation from Still Images | CVPR | 2021 | Event Frame Encoding | [GitHub](https://github.com/lyuchenyang/ESS) |
| Combining Events and Frames via Recurrent Asynchronous Networks | ECCV | 2020 | RNN + Fusion | [arXiv](https://arxiv.org/abs/2003.07547) |
| EV-SegNet: Asynchronous Event Segmentation Network | ICCV | 2023 | Hybrid Stream Fusion | [arXiv](https://arxiv.org/abs/2303.13684) |

#### 💡 Notes:
- Event cameras output sparse, high-frequency signals. Fusion methods include **event frame accumulation**, **voxelization**, and **cross-attention with RGB**.
- Event-based fusion is especially beneficial in **HDR** or **fast-motion scenes** where RGB degrades.

---
## 🔄 4. Adaptation Challenges in Multimodal Learning

### 4.1 Modality Adaptation Segmentation

| Paper Title | Venue | Year | Key Idea | Link |
|-------------|-------|------|----------|------|
| Achieving Cross Modal Generalization with Multimodal Unified Representation | NeurIPS | 2022 | Unified Representation for Modal Transfer | [arXiv](https://arxiv.org/abs/2209.15113) |
| Unsupervised Modality Adaptation with Text-to-Image Diffusion | ICCV | 2023 | Diffusion-guided Modality Synthesis | [arXiv](https://arxiv.org/abs/2303.08752) |
| Modality-Aware Knowledge Distillation for RGB-Thermal Tasks | ECCV | 2022 | Cross-modal Teacher-Student Training | [arXiv](https://arxiv.org/abs/2203.01970) |

#### 💡 Notes:
- The goal is to transfer knowledge between different modalities (e.g. RGB→TIR).


---

### 4.2 Missing Modalities

| Paper Title | Venue | Year | Key Idea | Link |
|-------------|-------|------|----------|------|
| Centering the Value of Every Modality: Efficient Modality-Agnostic Segmentation | CVPR | 2023 | Conditional Prompt Fusion | [arXiv](https://arxiv.org/abs/2304.04277) |
| Learning Modality-Agnostic Representation for Semantic Segmentation | NeurIPS | 2022 | Random Modality Dropout + Alignment | [arXiv](https://arxiv.org/abs/2211.11656) |
| Robust Multimodal Learning with Missing Modalities | ICLR | 2023 | Adapter-based Modality Plug-and-Play | [arXiv](https://arxiv.org/abs/2302.03286) |

#### 💡 Notes:
- The design goal is to ensure that the model can effectively predict even if a mode is missing.
---

### 4.3 Cross-Domain & Cross-Modal Domain Adaptation

| Paper Title | Venue | Year | Key Idea | Link |
|-------------|-------|------|----------|------|
| Cross-Domain and Cross-Modal Knowledge Distillation for 3D Segmentation | CVPR | 2022 | KD from Source RGB-LiDAR to Target | [arXiv](https://arxiv.org/abs/2203.05906) |
| Sparse-to-Dense Feature Matching in Cross-Modal Domain Adaptation | ECCV | 2022 | Multi-Level Alignment for RGB-LiDAR | [arXiv](https://arxiv.org/abs/2208.01952) |
| VFM-DA: Vision Foundation Models for Cross-Modal DA | ICCV | 2023 | Foundation Model Pretraining + Adaptation | [arXiv](https://arxiv.org/abs/2304.04513) |
| Towards Source-Free Domain Adaptive Semantic Segmentation Via Importance-Aware and Prototype-Contrast Learning | IEEE | 2024 | Importance-Aware + Prototype-Contrast | [arXiv](https://arxiv.org/abs/2306.01598) |

#### 💡 Notes:
- There is both **modality change** and **domain change** (e.g., RGB-D from indoor→outdoor).






## 📢 Contributions

Pull requests are welcome to expand this resource with new datasets, benchmarks, and literature. Feel free to fork this repo and help build a community resource.

---

## 📄 License

Open for academic and educational use. Please cite original datasets and papers when using any content listed here.
