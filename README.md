# X-INF554 Kaggle Data Challenge

Team: GWG

Members: Chenwei WAN, Xianjin GONG, Mengfei GAO

## How to use with pre-computed embeddings

Download utterance embeddings pre-computed by our tuned RoBERTa-large model:

| Binary Package                                                                                     | Size    |
|----------------------------------------------------------------------------------------------------|---------|
| [Train-dev](https://drive.google.com/file/d/1RY5aRUCezixco-Xy1rBVMjORnoIPuPLH/view?usp=drive_link) | 284.6MB |
| [Test](https://drive.google.com/file/d/13WViF-bhfJd69fGw_wkuktrpiu8yF4nq/view?usp=drive_link)      | 121.6MB |

Put these two `.pkl` files under `data/encoding`.

Run the following command for cross-validation:

```bash
python main.py -s pre-encoded-gcn -m train
```

For testing and producing the output file:

```bash
python main.py -s pre-encoded-gcn -m test
```

The model will firstly be trained on the whole training data and then make predictions on the test set.

## How to tune RoBERTa-large

Put raw data including utterances, graph edges and training labels under `data/`.

Download [RoBERTa-large](https://huggingface.co/roberta-large/tree/main) and put the files
under `modules/roberta-large`.

Run the following command to perform cross-validation (optional):

```bash
python main.py -m tune
```

Run the following command to tune RoBERTa-large for specified epochs and compute utterance embeddings save as binary
packages:

```bash
python main.py -m compute -e [epochs-to-tune]
```