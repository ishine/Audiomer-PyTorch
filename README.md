<div align="center">


<samp>


# ☢️ Audiomer ☢️ 

</samp>
  
Audiomer: A Convolutional Transformer for Keyword Spotting
<br>
Accepted at AAAI 2022 DSTC Workshop 
  
| **[ [```arXiv```](<https://arxiv.org/abs/2109.10252>) ]** | **[ [```Previous SOTA```](<https://github.com/ARM-software/keyword-transformer>) ]** |**[ [```Model Architecture```](<assets/Audiomer.jpeg>) ]** |
|:-------------------:|:-------------------:|:-------------------:|

<br>
<samp>

### Pretrained Models: [Google Drive](<https://drive.google.com/drive/folders/1yw2Rc84e6xgBteMYPIL1ny9XELnve3FX?usp=sharing>)
### NOTE: This is a pre-print release, the code might have bugs.

## Usage
To reproduce the results in the paper, follow the instructions:

- To download the Speech Commands v2 dataset, run: `python3 datamodules/SpeechCommands12.py`
- To train Audiomer-S and Audiomer-L on all three datasets thrice, run: `python3 run_expts.py`
- To evaluate a model on a dataset, run: `python3 evaluate.py --checkpoint_path /path/to/checkpoint.ckpt --model <model type> --dataset <name of dataset>`.
- For example: `python3 evaluate.py --checkpoint_path ./epoch=300.ckpt --model S --dataset SC20`

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
