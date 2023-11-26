from utils.training_naive_roberta import train_naive_roberta, test_naive_roberta
from utils.training_gcn_roberta import train_gcn_roberta, test_gcn_roberta

MODELS = {
    "naive-roberta": {
        "train": train_naive_roberta,
        "test": test_naive_roberta
    },
    "gcn-roberta": {
        "train": train_gcn_roberta,
        "test": test_gcn_roberta
    }
}
