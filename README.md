# LIST-Diffusion

This repository contains the code for our paper:

**LIST-Diffusion: LLM-enhanced Interpretable Structured Text-driven Diffusion Model for Abnormal Traffic Situation Generation**



## Highlights

  LIST-Diffusion is a knowledge- and text-driven traffic situation generation model. It leverages large language models (LLMs) to enrich short traffic event descriptions and guides a time-aware latent diffusion model to generate realistic and interpretable abnormal traffic scenarios. The model captures how congestion forms, propagates, and interacts with the road network, providing knowledge-driven, semantically meaningful traffic simulation.



## Usage

### Datasets

Download the dataset from [BjTT: A Large-scale Multimodal Dataset for Traffic Prediction](https://chyazhang.github.io/BjTT)  
Organize the files as follows:
```
|-- your dataset root dir/
|   |-- traffic/
|       |-- train
|            |-- data
|            |-- text
|       |-- validation
|            |-- data
|            |-- text
|       |-- matrix_roadclass&length.npy
|       |-- matrix.npy
|       |-- Roads1260.json
|       |-- train.txt
|       |-- validation.txt
```

### Requirements

Create a conda environment and install dependencies:

```bash
conda create -n LIST-Diffusion python=3.8.18
conda activate LIST-Diffusion
pip install -r environment.yaml
```

### Training

Stage 1: Autoencoder Pretraining
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/autoencoder/autoencoder_traffic.yaml -t --gpus 1
```
Stage 2: LIST-Diffusion Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/LIST-Diffusion/traffic.yaml -t --gpus 1
```
Note: Adjust `--gpus` according to the number of GPUs available.

### Testing

Generate a traffic scenario using a textual prompt:
```bash
python scripts/LIST-Diffusion.py --prompt "January 01, 2022, 08:08. road closure on Wufang Bridge. construction and road closure on Jingliang Road. road closure on Tuanhe Road. construction and road closure on South Third Ring West Road..."
```
### Visualization

You can visualize the generated traffic maps using the built-in script:
```bash
python scripts/plot_map.py
```
The visualization supports color-coded congestion levels and road network overlay.

## Acknowledgments

This code is built upon [Latent Diffusion](https://github.com/CompVis/latent-diffusion).

## BibTeX

If you find this work useful, please cite:
```
@inproceedings{lu2025listdiffusion,
  title={LIST-Diffusion: LLM-enhanced Interpretable Structured Text-driven Diffusion Model for Abnormal Traffic Situation Generation},
  author={Yaxuan Lu, Guangyu Huo*, Xiaohui Cui, Boyue Wang, Yong Zhang, Zhiyong Cui},
  booktitle={...},
  year={2025}
}
```

## Contact

For questions or issues, please contact:
Yaxuan Lu – luyaxuan@bjfu.edu.cn

