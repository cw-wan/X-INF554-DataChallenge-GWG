# X-INF554 Kaggle Data Challenge

## How to use:

1. Download data to `data/`
2. Download [RoBERTa](https://huggingface.co/roberta-base) to `modules/roberta-base`. 
3. For training, run command `python main.py -s [model] -m train`.
4. To generate output csv file, run command `python main.py -s [model] -m test -e [checkpoint_epoch]`

## Model list

1. naive-roberta
2. gcn-roberta
3. gat-roberta