# SpatialReasoner

Official implementation of **SpatialReasoner**, from the following paper

[SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning](https://spatial-reasoner.github.io/).<br/>
[Wufei Ma*](https://wufeim.github.io/), [Yu-Cheng Chou*](https://sites.google.com/view/yu-cheng-chou), [Qihao Liu*](https://qihao067.github.io/), [Xingrui Wang](https://xingruiwang.github.io/), [Celso de Melo†](https://celsodemelo.net/), [Jieneng Chen](https://beckschen.github.io/), [Jianwen Xie^](http://www.stat.ucla.edu/~jxie/), and [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<br/>
Johns Hopkins University, †DEVCOM Army Research Laboratory, ^Lambda Inc<br/>
[[`arXiv`](http://arxiv.org/abs/2504.20024)] [[`Project Page`](https://spatial-reasoner.github.io/)]

<p align="center">
    <img src="assests/motivation_thinking.png"/> <br />
    <em> 
    Comparing 3D spatial reasoning of our SpatialReasoner with previous state-of-the-art models. Our SpatialReasoner builds on explicit 3D representations, performs 3D computation, and reasons about the final answer. Although Gemini 2.0 can also break down complex 3D spatial reasoning questions into small and tractable steps, it lacks reliable 3D computation that leads to the correct answer.
    </em>
</p>

## Installation

Setup Python dependencies.

```bash
conda create -n spatial_reasoner python=3.11 -y && conda activate spatial_reasoner
pip3 install -e ".[dev]"
pip3 install flash-attn --no-build-isolation
pip3 install qwen_vl_utils xlsxwriter
```

Setup evaluation *environment.*

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


<p align="center">
    <img src="assests/benchmark.png"/> <br />
    <em> 
    Comparison with previous state-of-the-art methods on 3DSRBench. Our SpatialReasoner outperforms previous open-source and proprietary methods on challenging 3D spatial reasoning problems in 3DSRBench.
    </em>
</p>

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
