## Unifying Generation and Compression: Ultra-low bitrate Image Coding Via Multi-stage Transformer

> This is the official implementation of image compression based on the fine tuned VQGAN model.<br>
> [Naifu Xue](https://scholar.google.com/citations?user=WYqicbgAAAAJ&hl=zh-CN&oi=sra), [Qi Mao](https://scholar.google.com/citations?user=VTQZF6EAAAAJ&hl=zh-CN&oi=sra), Zijian Wang, [Hao Wei](https://github.com/cshw2021), Yuan Zhang, [Siwei Ma](https://scholar.google.com/citations?user=y3YqlaUAAAAJ&hl=zh-CN&oi=sra)<br>
> :partying_face: This work is accepted by 2024 IEEE International Conference on Multimedia and Expo (ICME).
<p align="center">
    <img src="assets/Framework.png" style="border-radius: 15px"><br>
</p>

## :book: Table Of Contents
- [:eyes: Visual Results](#visual_results)
- [:crossed\_swords: Quantitative Performance](#quantitative_performance)
- [:computer: Train](#computer-train)
- [:zap: Inference](#inference)
- [:memo: TODO](#todo)
- [:heart: Acknowledgement](#acknowledgement)
- [:clipboard: Citation](#cite)

## <a name="visual_results"></a>:eyes: Visual Results
<p align="center">
    <img src="assets/visual_results.png" style="border-radius: 15px"><br>
</p>

## <a name="quantitative_performance"></a>:crossed_swords: Quantitative Performance
<p align="center">
    <img src="assets/quantitative.png" style="border-radius: 15px"><br>
</p>

## :wrench: Requirements

```bash
- conda env create -f environment.yml
- conda activate UIGC
- pip install bitstream==2.6.0.2
```

## <a name="inference"></a>:zap: Inference
1. Download fine-tuned VQGAN from [Google Driver](https://drive.google.com/drive/folders/14I_RnQ3cA6etdKGPVMFdmmVgMtBTB5rn?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1zBeWKh6vgof13iTBwtA65A?pwd=kfl7) (code: kfl7) into `./pretrained`.
 
2. Download the pre-trained transformer from [Baidu Cloud](https://pan.baidu.com/s/1iHQed7QqfuPJwlGOlF12kQ?pwd=7lqt) (code:7lqt).

3. Here we provide three modes of reconstruction:
* ‘entire’ indicates the mode without any Mask applied,
* ’edge_more‘ refers to the mode where a checkerboard Mask is generated based on the edge map,
* ‘minium’ denotes the mode with a full checkerboard Mask.
* Run the following command. 

   ```
   python test.py --base configs/test/entire/Kodak/VQ16_Kodak.yaml --gpus 0,
   ```
   
## <a name="todo"></a>:memo: TODO
- [X] Release pretrained models.
- [X] Release inference code.
- [ ] Release training code.

## <a name="acknowledgement">:heart: Acknowledgement
This work is based on [VQGAN](https://github.com/lllyasviel/ControlNet), [miniGPT](https://github.com/karpathy/minGPT), thanks to their invaluable contributions.

## <a name="cite"></a>:clipboard: Citation

Please cite us if our work is useful for your research.

```
@article{xue2024unifying,
  title={Unifying Generation and Compression: Ultra-low bitrate Image Coding Via Multi-stage Transformer},
  author={Xue, Naifu and Mao, Qi and Wang, Zijian and Zhang, Yuan and Ma, Siwei},
  journal={arXiv preprint arXiv:2403.03736},
  year={2024}
}
```