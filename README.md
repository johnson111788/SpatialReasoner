# SpatialReasoner

Official implementation of **SpatialReasoner**, from the following paper

[SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning](https://spatial-reasoner.github.io/).<br/>
[Wufei Ma*](https://wufeim.github.io/), [Yu-Cheng Chou*](https://sites.google.com/view/yu-cheng-chou), [Qihao Liu*](https://qihao067.github.io/), [Xingrui Wang](https://xingruiwang.github.io/), [Celso de Melo](https://celsodemelo.net/), [Jianwen Xie](http://www.stat.ucla.edu/~jxie/), and [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<br/>
Johns Hopkins University<br/>
[[`arXiv`](http://arxiv.org/abs/2504.20024)] [[`Project Page`](https://spatial-reasoner.github.io/)]

![Motivation Thinking](assests/motivation_thinking.png)

## Installation

Setup Python dependencies.

```bash
conda create -n spatial_reasoner python=3.11 -y && conda activate spatial_reasoner
pip3 install -e ".[dev]"
pip3 install flash-attn --no-build-isolation
pip3 install qwen_vl_utils xlsxwriter
```

Setup evaluation environment.

```bash
git submodule update --init --recursive
cd VLMEvalKit
pip install -e .
```

## Training

### Download Training Data

```bash
mkdir ./data && cd ./data

# OpenImages
wget https://huggingface.co/datasets/ccvl/SpatialReasonerTrain/resolve/main/openimages.tar
tar -xvf openimages.tar

# LLaVA
wget https://huggingface.co/datasets/ccvl/SpatialReasonerTrain/resolve/main/llava.tar
tar -xvf llava.tar

cd ../
```

### Training
- SpatialReasoner-SFT
```bash
bash local_scripts/spatialreasoner-sft.sh
```

- SpatialReasoner-Zero
```bash
bash local_scripts/spatialreasoner-zero.sh
```

- SpatialReasoner
```bash
bash local_scripts/spatialreasoner.sh
```

## Evaluation

### Download Evaluation Data

```sh
cd ./data

# 3DSRBench
wget https://huggingface.co/datasets/ccvl/3DSRBench/resolve/main/3dsrbench_v1_vlmevalkit_circular.tsv

# CV-Bench-3D
wget https://huggingface.co/datasets/ccvl/SpatialReasonerEval/resolve/main/CV-Bench-3D.tsv
```

### Inference

- SpatialReasoner-SFT
```bash
bash local_scripts/infer_spatialreasoner-sft.sh
```

- SpatialReasoner-Zero
```bash
bash local_scripts/infer_spatialreasoner-zero.sh
```

- SpatialReasoner
```bash
bash local_scripts/infer_spatialreasoner.sh
```

Results for `CVBench3D` are printed to the terminal (stdout), and the final results for `3DSRBench` are saved to `results_3DSRBench.csv`.

## Citation

If you find this repository helpful, please consider citing:

```
@article{ma2025spatialreasoner,
  title={SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning},
  author={Ma, Wufei and Chou, Yu-Cheng and Liu, Qihao and Wang, Xingrui and de Melo, Celso and Xie, Jianwen and Yuille, Alan},
  journal={arXiv preprint arXiv:2504.20024},
  year={2025}
}
```
