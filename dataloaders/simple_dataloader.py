import json
from torch.utils.data import Dataset, DataLoader
import os


class Simple(Dataset):
    def __init__(self,
                 subset,
                 config):
        super().__init__()

        assert subset in ["train", "dev", "test"]

        self.data = []

        if subset in ["train", "dev"]:
            utt_labels = {}
            positive_ids = []
            negative_ids = []
            idx = 0
            # read label data
            labels_path = os.path.join(config.Path.data, "training_labels.json")
            with open(labels_path) as lf:
                training_labels = json.load(lf)
            for dialogue_id, labels in training_labels.items():
                utt_labels[dialogue_id] = []
                for label in labels:
                    utt_labels[dialogue_id].append(label)
            # read dialogue data
            training_path = os.path.join(config.Path.data, "training")
            for _, _, file_list in os.walk(training_path):
                for _, f in enumerate(file_list):
                    if f[-4:] == "json":
                        with open(os.path.join(training_path, f)) as jf:
                            utterances = json.load(jf)
                            for _id, u in enumerate(utterances):
                                utt = {
                                    "id": f.split(".")[0] + "_" + str(_id),
                                    "text": u["text"],
                                    "label": utt_labels[f.split(".")[0]][_id]
                                }
                                self.data.append(utt)
                                if utt["label"] == 1:
                                    positive_ids.append(idx)
                                else:
                                    negative_ids.append(idx)
                                idx += 1
            # split training and dev set
            dev_train_ratio = config.DownStream.dev_train_ratio
            dev_positive_cnt = int(len(positive_ids) * dev_train_ratio)
            train_positive_cnt = len(positive_ids) - dev_positive_cnt
            dev_negative_cnt = int(len(negative_ids) * dev_train_ratio)
            train_negative_cnt = len(negative_ids) - dev_negative_cnt
            train_positive_ids = positive_ids[:train_positive_cnt]
            train_negative_ids = negative_ids[:train_negative_cnt]
            # Augment
            if config.DownStream.resample:
                train_positive_ids = train_positive_ids + train_positive_ids[:]
            dev_positive_ids = positive_ids[-dev_positive_cnt:]
            dev_negative_ids = negative_ids[-dev_negative_cnt:]
            if subset == "train":
                self.data = [self.data[_id] for _id in train_positive_ids + train_negative_ids]
            else:
                self.data = [self.data[_id] for _id in dev_positive_ids + dev_negative_ids]
        else:
            test_path = os.path.join(config.Path.data, "test")
            for _, _, file_list in os.walk(test_path):
                for idx, f in enumerate(file_list):
                    if f[-4:] == "json":
                        with open(os.path.join(test_path, f)) as jf:
                            utterances = json.load(jf)
                            for _id, u in enumerate(utterances):
                                utt = {
                                    "id": f.split(".")[0] + "_" + str(_id),
                                    "text": u["text"],
                                }
                                self.data.append(utt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def simple_dataloader(subset, config, batch_size, shuffle=True):
    dataset = Simple(subset, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
