import os
import json
from torch.utils.data import Dataset, DataLoader


def read_dialogues(path):
    utterances = {}
    for root, _, file_list in os.walk(path):
        for f in file_list:
            dialogue_id = f.split(".")[0]
            if f[-4:] == "json":
                with open(os.path.join(root, f), "r") as jf:
                    us = json.load(jf)
                    for u in us:
                        utterance_id = dialogue_id + "_" + str(u["index"])
                        utterances[utterance_id] = {
                            "id": utterance_id,
                            "text": u["text"],
                            "label": -1,
                        }
    return utterances


class TunerDataset(Dataset):
    def __init__(self, split, config, kth=None):
        super(TunerDataset, self).__init__()
        assert split in ["train", "dev", "train-dev", "test"]

        self.data = []

        if split in ["train", "dev", "train-dev"]:
            # read training dialogues
            utterances = read_dialogues(os.path.join(config.Path.data, "training"))
            # read training labels
            with open(os.path.join(config.Path.data, "training_labels.json")) as lf:
                training_labels = json.load(lf)
                for dialogue_id, labels in training_labels.items():
                    for idx, label in enumerate(labels):
                        utterance_id = dialogue_id + "_" + str(idx)
                        utterances[utterance_id]["label"] = label
            self.data = list(utterances.values())
            if split in ["train", "dev"]:
                # simple K-folds cross validation
                dev_size = len(self.data) // config.DownStream.folds
                assert 0 <= kth < config.DownStream.folds
                dev_start = kth * dev_size
                dev_end = (kth + 1) * dev_size
                if split == "train":
                    del self.data[dev_start:dev_end]
                else:
                    self.data = self.data[dev_start:dev_end]
        elif split == "test":
            # read testing dialogues
            utterances = read_dialogues(os.path.join(config.Path.data, "test"))
            self.data = list(utterances.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tuner_dataloader(batch_size, shuffle, split, config, kth=None):
    dataset = TunerDataset(split, config, kth)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
