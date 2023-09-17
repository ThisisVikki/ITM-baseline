# ITM-baseline: IRNet

### Lightweight Improved Residual Network for Efficient Inverse Tone Mapping
[paper link](https://arxiv.org/abs/2307.03998)

![image](https://github.com/ThisisVikki/ITM-baseline/blob/main/figure/Net%26block_structure_newcrop.png)
**Figure:** *Architecture of the proposed Improved Residual Network (IRNet) and the Improved Residual Block (IRB)*

## Getting Started
### Dataset
We use the [HDRTV1K](https://github.com/chxy95/HDRTVNet#configuration) dataset for training, and the test set of [HDRTV1K](https://github.com/chxy95/HDRTVNet#configuration), the test set of [Deep SR-ITM](https://github.com/sooyekim/Deep-SR-ITM), and our ITM-4K test dataset for testing. The ITM-4K test dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1AJqz2ILR9XD_UO4lFklBOoKpj4ZOrRmo/view?usp=drive_link) and [Baidu NetDisk](https://pan.baidu.com/s/1KK9s6DU-ZAKIEydMnFfClQ) (access code: t7iy). It contains 160 pairs of SDR and HDR images of size $3840\times2160\times3$.

### How to test

### How to Train

## References
If our work is helpful for you, please cite our paper:
```BibTeX
@misc{xue2023lightweight,
      title={Lightweight Improved Residual Network for Efficient Inverse Tone Mapping}, 
      author={Liqi Xue and Tianyi Xu and Yongbao Song and Yan Liu and Lei Zhang and Xiantong Zhen and Jun Xu},
      year={2023},
      eprint={2307.03998},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
The code and the proposed test dataset is released for academic research only.
