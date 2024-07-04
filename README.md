# TFG: Generating Adversarial Examples for Misinformation detection using Automated Text Simplification

This repository contains scripts and instructions for setting up and running various experiments.

## Virtual Environment

### Create and Activate Virtual Environment
To create and activate a virtual environment, use the following commands:

```sh
# Activate
$ source venv/bin/activate

# Deactivate
$ deactivate

# Eliminate
$ rm -rf venv
```

## Generate the Datasets
To generate the datasets, run:
```sh
python ./conversion/convert_PR2.py ~/Desktop/TFG/BODEGA/datasets ~/Desktop/TFG/BODEGA/
```

## Training Victim Classifiers
To train the victim classifiers, run:
```sh
python ./runs/train_victims.py PR2 BiLSTM ~/Desktop/TFG/BODEGA/ ~/Desktop/TFG/BODEGA/bilstm.pth
```

## Testing the Attack Performance
To train the victim classifiers, run:
```sh
# Attack performance with RRW
python ./runs/attack.py PR2 true RRW BiLSTM ~/Desktop/TFG/BODEGA ~/Desktop/TFG/BODEGA/bilstm.pth

# Attack performance with BERTattack
python ./runs/attack.py PR2 true BERTattack BiLSTM ~/Desktop/TFG/BODEGA ~/Desktop/TFG/BODEGA/bilstm.pth

# Attack performance on HN with Simplification
python ./runs/attack.py HN true Simplification BiLSTM ~/Desktop/TFG/BODEGA ~/Desktop/TFG/BODEGA/bilstm.pth

# Attack performance on HN with RRW
python ./runs/attack.py HN true RRW BiLSTM ~/Desktop/TFG/BODEGA ~/Desktop/TFG/BODEGA/datasets/HN/bilstm.pth

# Attack performance on FC with RRW
python ./runs/attack.py FC true RRW BiLSTM ~/Desktop/TFG/BODEGA ~/Desktop/TFG/BODEGA/bilstm.pth

```

## LAMBO (Split Text into Sentences)
To train the victim classifiers, run:
```sh
$ source venv/bin/activate

# Split sentences
python /Users/martinagomez/Desktop/TFG/BODEGA/split/split_sentences PR2

```

## MUSS (Perform Sentence Simplification)
MUSS requires Python 3.8.

```sh
# Activate virtual environment
$ source myenv/bin/activate
cd muss

# Simplify sentences
python scripts/simplify.py split/output/output (path to sentences) --model-name muss_en_wikilarge_mined
python scripts/simplify.py /Users/martinagomez/Desktop/TFG/BODEGA/split/output/output.txt --model-name muss_en_wikilarge_mined

# Simplify tasks
python simplify/simplify_sentences.py HN --model-name muss_en_wikilarge_mined

```

### To Do
Install fairseq from Facebook research: fairseq GitHub repository

