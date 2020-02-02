# Sementics-Textual-Semilarity
The semantics textual similarity means textual similar between two sentence such as ``` i am a good boy``` and ``` i am not a bad boy``` those two sentence are same meaning. so i try to find out those sentences are same meaning or not. i use public open souces data that have three class category that are neutral,contradiction and entailment. 

# Requirements
- python 2.7
- tensorflow 1.5

# Data format
the train and test sets require a tab-separated format. Each line in the train (or test) file corresponds to an instance, and it should be arranged as

label sentence#1 sentence#2 instanceID

For more details about the data format, you can download the SNLI and the Quora Question Pair datasets.

# Training
```
python train.py --config_path configs/snli.sample.config
```

# Testing
```
python src/Decoder.py --in_path data/snli/dev.tsv --word_vec_path data/snli/wordvec.txt --out_path result.txt --model_prefix logs/SentenceMatch.snli
```
