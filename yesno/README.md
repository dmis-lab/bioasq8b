# yesno

## Requirements
```bash
conda create -n bioasq_yesno python=3.7
conda activate biopasq_yesno
conda install numpy tqdm
conda install pytorch=1.5.1 cudatoolkit=10.1 -c pytorch
pip install transformers==2.11.0
```

## Train

The training will finish in about 3 minutes

```bash
export BIOBERT_DIR=$HOME/BioASQ/pt_biobert1.1_mnli
export BIOASQ_DIR=$HOME/BioASQ/data-release
python run_yesno.py \
    --model_name_or_path $BIOBERT_DIR \
    --do_train \
    --overwrite_cache \
    --train_file $BIOASQ_DIR/BioASQ-7b/train/Snippet-as-is/BioASQ-train-yesno-7b-snippet.json \
    --train_batch_size 24 \
    --learning_rate 8e-6 \
    --num_train_epochs 3.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 10 \
    --seed 0 \
    --output_dir /tmp/yesno_output
```

## Evaluate

### Prediction in SQuAD format

The prediction output is saved in "predictions_.json" This needs to be tranformed in BIOASQ format for the official evaluation module.

```bash
export MODEL_DIR=/tmp/yesno_output
export BIOASQ_DIR=$HOME/BioASQ/data-release
python run_yesno.py \
    --model_name_or_path $MODEL_DIR \
    --overwrite_cache \
    --do_predict \
    --predict_file $BIOASQ_DIR/BioASQ-7b/test/Snippet-as-is/BioASQ-test-yesno-7b-1-snippet.json \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /tmp/yesno_output
```

### Evaluation in BioASQ format

```bash
export OUTPUT_DIR=/tmp/yesno_output
export Evaluation_Measures_DIR=$HOME/Evaluation-Measures/
export BIOASQ_DIR=$HOME/BioASQ/data-release
python ./transform_n2b_yesno.py \
    --nbest_path=$OUTPUT_DIR/predictions_.json \
    --output_path=$OUTPUT_DIR
java -Xmx10G \
    -cp $CLASSPATH:$Evaluation_Measures_DIR/flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
    evaluation.EvaluatorTask1b \
    -phaseB \
    -e 5     \
    $BIOASQ_DIR/BioASQ-7b/test/Golden/7B1_golden.json     \
    $OUTPUT_DIR/BioASQform_BioASQ-answer.json
```

### Result

```bash
0.896551724137931 0.0 0.0 0.0 0.0 0.0 0.0 0.8317214700193424 0.9361702127659575 0.7272727272727273
```