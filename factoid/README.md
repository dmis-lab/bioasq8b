# factoid

## Requirements
```bash
conda create -n bioasq_factoid python=3.7
conda activate biopasq_factoid
pip install tensorflow-gpu==1.14.0
pip install tqdm
```

## Fine-tuning
Please use `run_factoid.py` for factoid type questions. Use `BioASQ-*.json` for training and predicting dataset.
Our pre-processed version of the BioASQ Task 8b-Phase B dataset is available at **[`Pubmed Abstract`](https://drive.google.com/drive/folders/1SlgDQUg2hNMBRDgPZlqo_ucRZpDM3TV6)** and **[`Factoid type`](https://drive.google.com/drive/folders/1SlgDQUg2hNMBRDgPZlqo_ucRZpDM3TV6)**.
Follow the below processes to train and predict for our model.

### Training and Predicting
If you want to predict then set the `--do_predict` argument as True else False.

```bash
export BIOBERT_DIR=Directory of the Pre-trained Language Model
export DATA_DIR=Directory of the pre-processed version of BioASQ dataset
export batch=A number of testset batch (e.g., 1~5)

python run_factoid.py \
    --do_train=True \
    --do_predict=True \
    --vocab_file=$(BIOBERT_DIR)/vocab.txt \
    --bert_config_file=$(BIOBERT_DIR)/config.json \
    --init_checkpoint=$(BIOBERT_DIR)/model.ckpt \
    --max_seq_length=384 \
    --train_batch_size=24 \
    --learning_rate=5e-6 \
    --batch=$(batch) \
    --doc_stride=128 \
    --num_train_epochs=3.0 \
    --do_lower_case=False \
    --train_file=$(DATA_DIR)/train_v2/Snippet-as-is/BioASQ-train-factoid-8b-snippet-annotated.json \
    --predict_file=$(DATA_DIR)/test/Snippet-as-is/BioASQ-test-factoid-8b-$(batch)-snippet.json \
    --output_dir=Directory of output file \
```

### Evaluation
We already have the evaluation code built-in. However, this part suggests only for the evaluation part.
The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in the `output_dir`.
Run transform file (e.g., `transform_n2b_factoid.py`) in `./biocodes/` folder to convert `nbest_predictions.json` or `predictions.json` to BioASQ JSON format, which will be used for the official evaluation.

```bash
python ./biocodes/transform_n2b_factoid.py \
    --nbest_path={OUTPUT_DIR}/nbest_predictions.json \
    --output_path={OUTPUT_DIR}
```

This will generate `BioASQform_BioASQ-answer.json` in {OUTPUT_DIR}.
Clone [Evaluation](https://github.com/BioASQ/Evaluation-Measures) code from BioASQ github and run evaluation code on `Evaluation-Measures` directory.
Please note that you should put 5 as parameter for -e if you are evaluating the system for BioASQ 5b/6b/7b/8b dataset.

```bash
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    $(DATA_DIR)/test/7B1_golden.json \
    $(OUTPUT_DIR)/BioASQform_BioASQ-answer.json | cut -d' ' -f2,3,4 | sed -e 's/ /\t/g'
```

This will give you the below scores.
The evaluation scores are sequentially recurs to SAcc, LAcc, MRR.
```bash
0.00 0.00 0.00
```
