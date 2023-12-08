from utils.training_pre_encoded_gcn import train_pre_encoded_gcn, test_pre_encoded_gcn

MODELS = {
    "pre-encoded-gcn": {
        "train": train_pre_encoded_gcn,
        "test": test_pre_encoded_gcn
    }
}
