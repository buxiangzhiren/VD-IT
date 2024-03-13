[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

The official implementation of the paper: 

<div align="center">
<h1>
<b>
Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation
</b>
</h1>
</div>

<p align="center"><img src="docs/VD-IT.PNG" width="800"/></p>

> [**Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation**]()
>
> Zixin Zhu, Xuelu Feng, Dongdong CHen, Junsong Yuan, Chunming Qiao, Gang Hua

### Abstract

  In this paper, we explore the visual representations produced from a pre-trained text-to-video (T2V) diffusion model for video understanding tasks. We hypothesize that the latent representation learned from a pretrained generative T2V model encapsulates rich semantics and coherent temporal correspondences, thereby naturally facilitating video understanding. Our hypothesis is validated through the classic referring video object segmentation (R-VOS) task. We introduce a novel framework, termed ``VD-IT'', tailored with dedicatedly designed components built upon a fixed pretrained T2V model. Specifically, VD-IT uses textual information as a conditional input, ensuring semantic consistency across time for precise temporal instance matching. It further incorporates image tokens as supplementary textual inputs, enriching the feature set to generate detailed and nuanced masks.Besides, instead of using the standard Gaussian noise, we propose to predict the video-specific noise with an extra noise prediction module, which can help preserve the feature fidelity and elevates segmentation quality. Through extensive experiments, we surprisingly observe that fixed generative T2V diffusion models, unlike commonly used video backbones (e.g., Video Swin Transformer) pretrained with discriminative image/video pre-tasks, exhibit better potential to maintain semantic alignment and temporal consistency. On existing standard benchmarks, our VD-IT achieves highly competitive results, surpassing many existing state-of-the-art methods. 



[//]: # (## Requirements)

[//]: # ()
[//]: # (We test the codes in the following environments, other versions may also be compatible:)

[//]: # ()
[//]: # (- CUDA 11.1)

[//]: # (- Python 3.7)

[//]: # (- Pytorch 1.8.1)

[//]: # ()
[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (Please refer to [install.md]&#40;docs/install.md&#41; for installation.)

[//]: # ()
[//]: # (## Data Preparation)

[//]: # ()
[//]: # (Please refer to [data.md]&#40;docs/data.md&#41; for data preparation.)

[//]: # ()
[//]: # (We provide the pretrained model for different visual backbones. You may download them [here]&#40;https://drive.google.com/drive/u/0/folders/11_qps3q75aH41IYHlXToyeIBUKkfdqso&#41; and put them in the directory `pretrained_weights`.)

[//]: # ()
[//]: # (<!-- For the Swin Transformer and Video Swin Transformer backbones, the weights are intialized using the pretrained model provided in the repo [Swin-Transformer]&#40;https://github.com/microsoft/Swin-Transformer&#41; and [Video-Swin-Transformer]&#40;https://github.com/SwinTransformer/Video-Swin-Transformer&#41;. For your convenience, we upload the pretrained model in the google drives [swin_pretrained]&#40;https://drive.google.com/drive/u/0/folders/1QWLayukDJYAxTFk7NPwerfso3Lrx35NL&#41; and [video_swin_pretrained]&#40;https://drive.google.com/drive/u/0/folders/19qb9VbKSjuwgxsiPI3uv06XzQkB5brYM&#41;. -->)

[//]: # ()
[//]: # ()
[//]: # (After the organization, we expect the directory struture to be the following:)

[//]: # ()
[//]: # (```)

[//]: # (ReferFormer/)

[//]: # (├── data/)

[//]: # (│   ├── ref-youtube-vos/)

[//]: # (│   ├── ref-davis/)

[//]: # (│   ├── a2d_sentences/)

[//]: # (│   ├── jhmdb_sentences/)

[//]: # (├── davis2017/)

[//]: # (├── datasets/)

[//]: # (├── models/)

[//]: # (├── scipts/)

[//]: # (├── tools/)

[//]: # (├── util/)

[//]: # (├── pretrained_weights/)

[//]: # (├── eval_davis.py)

[//]: # (├── main.py)

[//]: # (├── engine.py)

[//]: # (├── inference_ytvos.py)

[//]: # (├── inference_davis.py)

[//]: # (├── opts.py)

[//]: # (...)

[//]: # (```)

[//]: # ()
[//]: # (## Model Zoo)

[//]: # ()
[//]: # (All the models are trained using 8 NVIDIA Tesla V100 GPU. You may change the `--backbone` parameter to use different backbones &#40;see [here]&#40;https://github.com/wjn922/ReferFormer/blob/232b4066fb7d10845e4083e6a5a2cc0af5d1757e/opts.py#L31&#41;&#41;.)

[//]: # ()
[//]: # (**Note:** If you encounter the `OOM` error, please add the command `--use_checkpoint` &#40;we add this command for Swin-L, Video-Swin-S and Video-Swin-B models&#41;.)



### Ref-Youtube-VOS & Ref-DAVIS17

|   Dataset    | J&F | J | F |
|:------------:| :----: | :----: | :----: |
|  Ref-Youtube-VOS   | 64.8  | 63.1  | 66.6 | 
|  Ref-DAVIS17   | 63.0  | 59.9  | 66.1 | 


### A2D-Sentences & JHMDB-Sentences

| Dataset | Overall IoU | Mean IoU | mAP  |
| :----: | :----: | :----: | :----: | |
| A2D-Sentences | 81.5 | 73.2 | 61.4 |
| JHMDB-Sentences | 74.4  | 73.4 | 46.5  |


### RefCOCO/+/g

We also support evaluate on RefCOCO/+/g validation set by using the pretrained weights (num_frames=1).


RIS (referring image segmentation):

| RefCOCO | RefCOCO+ | RefCOCOg |
|:-------:|:--------:|:--------:| 
|  76.7   |   66.5   |   70.3   |



## Acknowledgement

This repo is based on [ReferFormer](https://github.com/wjn922/ReferFormer) and [ModelScopeT2V](https://modelscope.cn/models/damo/text-to-video-synthesis/summary). Thanks for their wonderful works.


## Citation

```

```

