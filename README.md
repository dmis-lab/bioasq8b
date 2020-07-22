# Transferability of Natural Language Inference to Biomedical Question Answering
This repository provides a code for our paper, which tries to interpret the transferability of natural language inference to biomedical question answering.
We use this code for the [BioASQ](http://bioasq.org/) Challenge Task 8b-Phase B.
Please refer to our paper [Transferability of Natural Language Inference to Biomedical Question Answering](https://arxiv.org/abs/2007.00217) for more details.
This project is proceeded by [DMIS-Lab](https://dmis.korea.ac.kr).

## Data Download
We provide a pre-processed version of the [BioASQ](http://participants-area.bioasq.org/datasets/) Task 8b-Phase B
* **[`Pubmed Abstract`](https://drive.google.com/drive/folders/1JPMC4P7dgeoG-JW3yMKY8t6bnym5-GNb?usp=sharing)** : a pre-processed version of pubmed abstract data used for Task 8b-Phase B.
* **[`SQuAD Oracle`](https://drive.google.com/drive/folders/1FlEPhxr_2Y-WwpvepkTOOVvMd0aQ9n0x)** : a pre-processed version of SQuAD oracle dataset.
* **[`Yes/No type`](https://drive.google.com/file/d/1TnGPE5HQeVFoOn7hPAHemOLdVNgKut1J/view?usp=sharing)** : a pre-processed version of Yes/No type questions in Task 8b-Phase B.
* **[`Factoid type`](https://drive.google.com/file/d/1eP7w9iCZ84KuwRbwfbYCC3K1UPpyEe0f/view?usp=sharing)** : a pre-processed version of Factoid type questions in Task 8b-Phase B.
* **[`List type`](https://drive.google.com/file/d/1qpUvMosQ8ufIfuyeyKcVlsyEa7J0H7PY/view?usp=sharing)** : a pre-processed version of List type questions in Task 8b-Phase B.

We revised the pre-processed datasets from **Pre-trained Language Model for Biomedical Question Answering** released by [BioASQ-BioBERT](https://github.com/dmis-lab/bioasq-biobert).

For details of the original BioASQ datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.

## Pre-trained Model
We use the [BioBERT](https://github.com/dmis-lab/biobert) model as our base model learning.
For specific fine-tuning procedure, please see our corresponding folder respectively.

In order to facilitate the reproduction of our results, we provide our learned parameters of our model.
* **[`BioBERT-MNLI`](https://drive.google.com/drive/folders/16vzhUwGc88d3u-KjpiXB2KUik_UzRWuX)** : Sequential transfer learning of our model (BioBERT-MNLI) parameters. 
* **[`BioBERT-MNLI-SQuAD`](https://drive.google.com/drive/folders/1VKnhpPjGZja3jyYL2tZRP8ngdJnBJC8U)** : Sequential transfer learning of our model (BioBERT-MNLI-SQuAD) parameters.
* **[`BioBERT-MNLI-SQuAD(oracle)`](https://drive.google.com/drive/folders/1JB90k-FiAhqWlF_KD2jtKc0keWi2RDHh)** : Sequential transfer learning of our model (BioBERT-MNLI-SQuAD(oracle)) parameters.

## Requirements

* GPU (Our setting was a single NVIDIA Titan RTX (24GB) GPU)
* Python version >= 3.7
* Tensorflow version >= 1.14.0
* Pytorch version >= 1.5.1

## Contact Information
For help or any issues using our code, please contact Minbyul Jeong or Mujeen Sung (minbyuljeong, mujeensung {at} korea.ac.kr).
We welcome for any suggestions to modify our issues.

## Bibliography
```bash
@misc{jeong2020transferability,
    title={Transferability of Natural Language Inference to Biomedical Question Answering},
    author={Minbyul Jeong and Mujeen Sung and Gangwoo Kim and Donghyeon Kim and Wonjin Yoon and Jaehyo Yoo and Jaewoo Kang},
    year={2020},
    eprint={2007.00217},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
