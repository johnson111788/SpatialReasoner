# SpatialReasoner


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
