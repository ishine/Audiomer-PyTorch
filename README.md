<div align="center">


<samp>


# Audiomer 

</samp>
  
Audiomer: A Convolutional Transformer for Keyword Spotting
<br>
Accepted at AAAI 2022 DSTC Workshop 
  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/audiomer-a-convolutional-transformer-for-1/keyword-spotting-on-google-speech-commands)](https://paperswithcode.com/sota/keyword-spotting-on-google-speech-commands?p=audiomer-a-convolutional-transformer-for-1)  
  
| **[ [```arXiv```](<https://arxiv.org/abs/2109.10252>) ]** | **[ [```Previous SOTA```](<https://github.com/ARM-software/keyword-transformer>) ]** |**[ [```Model Architecture```](<assets/Audiomer.png>) ]** |
|:-------------------:|:-------------------:|:-------------------:|

<br>
<samp>
</div>
  
### Pretrained Models
  Links: [Google Drive](<https://drive.google.com/drive/folders/1yw2Rc84e6xgBteMYPIL1ny9XELnve3FX?usp=sharing>)
  Note: `The pretrained models only work with commit 6270ca27de47fbfd0379c172bbc74e6a61f72176`, after which there has been breaking changes.


## Usage
To reproduce the results in the paper, follow the instructions:

- To download the Speech Commands v2 dataset, run: `python3 download_speechcommands.py`
- To train Audiomer-S and Audiomer-L on all three datasets thrice, run: `python3 run_expts.py`
- To evaluate a model on a dataset, run: `python3 evaluate.py --checkpoint_path /path/to/checkpoint.ckpt --model <model type> --dataset <name of dataset>`.
- For example: `python3 evaluate.py --checkpoint_path ./epoch=300.ckpt --model S --dataset SC20`

<div align="center">

## Results 
<img src="assets/results.png">

## Performer Conv-Attention
TLDR: We augment 1D ResNets With Performer Attention over Raw Audio waveform. 

<img src="assets/ConvAttention.png">

---  
  
</div>

## System requirements
- NVIDIA GPU with CUDA
- Python 3.6 or higher.
- pytorch_lightning
- torchaudio
- performer_pytorch
