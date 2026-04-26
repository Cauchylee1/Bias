# BiasAlert

## Introduction

**BiasAlert** is a plug-and-play system designed for detecting social bias in text, powered by Retrieval-Augmented Generation (RAG). By integrating external knowledge with the model’s internal reasoning capabilities, BiasAlert significantly enhances the identification of socially biased content. This approach is detailed in our EMNLP 2024 paper:  
> *BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs*  
You can read the full paper [here](https://arxiv.org/pdf/2407.10241).
![BiasAlert Architecture](images/biasalert_architecture.jpg)

This repository includes instructions for data preprocessing, model training using the LLaMA-Factory framework, inference generation, and evaluation.

---

## Overview

1. [Installation](#installation)  
2. [Data Processing](#step-1-data-processing)  
3. [Training](#step-2-training)  
4. [Inference & Evaluation](#inference--evaluation)  

---

## Installation

```bash
conda create -n biasalert python=3.10
conda activate biasalert

pip install -r requirements.txt

cd code/LLaMA-Factory
pip install -e .[metrics]
```

---

## Training

### Step 1: Data Processing

We provide the processing scripts used to generate the training data, using **RedditBias** as an example. The processing workflow includes:

1. Constructing an instruction-tuning dataset **without** retrieval augmentation.  
2. Retrieving the top-5 most relevant bias-related knowledge entries for each training example.  
3. Reconstructing the instruction-tuning dataset with the retrieved information.  

You can execute the full pipeline with:

```bash
cd BiasAlert
bash ./data/data_precessing/data_processing.sh
```

The processed dataset structure looks like:

```
precessed_redditbias/
├── train/
│   ├── race.json
│   ├── gender.json
│   └── ...
├── test/
│   ├── race.json
│   ├── gender.json
│   └── ...
```

The specific script descriptions and corresponding parameter explanations are as follows:

---

1. `instruction_generation.py`

```bash
python data_processing/instruction_generation.py --category <CATEGORY>
```

Generates base instruction-tuning samples for a specified bias category.

| Argument     | Description |
|--------------|-------------|
| `--category` | The bias category to generate data for, e.g., `religion`, `race`, `gender`, etc. |

---

2. `generate_passage_embeddings.py`

```bash
python data_precessing/generate_passage_embeddings.py \
  --model_name_or_path facebook/contriever \
  --output_dir embeddings \
  --passages ./retrieval/bias_doc.tsv \
  --shard_id 0 --num_shards 1
```

Encodes a TSV corpus of bias-related passages into dense vectors using a retriever model like `facebook/contriever`.

| Argument              | Description |
|-----------------------|-------------|
| `--model_name_or_path` | Name or path of the retriever model (e.g., `facebook/contriever`). |
| `--output_dir`        | Directory to save the generated passage embeddings. |
| `--passages`          | Path to a `.tsv` file containing bias-related passages. |
| `--shard_id`          | ID of the current shard for parallel processing (set `0` for single machine). |
| `--num_shards`        | Total number of shards (set `1` for single machine). |

---

3. `run_retrieval.py`

```bash
python run_retrieval.py \
  --data ./precessed_redditbias/train \
  --passages ./retrieval/bias_doc.tsv \
  --passages_embeddings "embeddings/*" \
  --output_dir ./retrieval_outputs \
  --model_name_or_path facebook/contriever \
  --n_docs 5
```

Retrieves top-k relevant bias-related knowledge entries for each input example and saves the results.

| Argument                | Description |
|-------------------------|-------------|
| `--data`                | Path to the instruction-formatted data to augment with retrieved passages. |
| `--passages`            | Path to the same `.tsv` passage file used during embedding. |
| `--passages_embeddings` | Glob path pointing to saved embeddings (e.g., `embeddings/*`). |
| `--output_dir`          | Directory where retrieval-augmented outputs will be saved. |
| `--model_name_or_path`  | Retriever model used for query encoding (`facebook/contriever`). |
| `--n_docs`              | Number of top documents to retrieve for each sample. |

---

### Step 2: Training

We used the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework to train **BiasAlert**. Specifically, we ran the 
```bash
cd code/LLaMA-Factory
bash ./examples/lora_multi_gpu/rag_multi_node.sh
```
script with customized configurations to load our processed dataset.

We used **LLaMA-2-Chat 7B** as the base model and adopted LoRA-based parameter-efficient fine-tuning (PEFT), which is well supported in the LLaMA-Factory framework.

---

## Inference & Evaluation

### Inference

Use the script [`./code/generate_responses.py`](./code/generate_responses.py) to perform inference with the trained BiasAlert model. It supports red-teaming and harmful content detection tasks.

Based on the path to the test set, run the `./data/data_precessing/data_processing.sh` script to perform relevant document retrieval.  
The processed data will be saved under `<EVAL_DATASET_PATH>`.  
Then, use this file as input to the `generate_responses.py` script to perform inference.

```bash
python ./code/generate_responses.py \
  --prompt <PROMPT_TEMPLATE> \
  --model <MODEL_PATH_OR_NAME> \
  --dataset <EVAL_DATASET_PATH> \
  [--save_path <SAVE_DIRECTORY>] \
  [--num_samples <NUM_SAMPLES>] \
  [--load_8bit] \
  [--keep_thoughts]
```

### Evaluation

If your dataset contains ground-truth labels, use [`./code/eval_metrics.py`](./code/eval_metrics.py) to evaluate prediction accuracy:

```bash
python ./code/eval_metrics.py \
  --file_path <PREDICTIONS_FILE_PATH> \
  --save_path <SAVE_RESULTS_TO>
```

The script computes evaluation metrics including **accuracy**, **precision**, **recall**, and **F1-score**.

---

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{fan2024biasalert,
  title={BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs},
  author={Fan, Zhiting and Chen, Ruizhe and Xu, Ruiling and Liu, Zuozhu},
  journal={arXiv preprint arXiv:2407.10241},
  year={2024},
  url={https://arxiv.org/pdf/2407.10241}
}
```
