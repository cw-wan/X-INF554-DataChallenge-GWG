import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import random

MAP_RELATIONS = {
    "Elaboration": 0, "Question-answer_pair": 1, "Correction": 2, "Contrast": 3, "Background": 4, "Explanation": 5,
    "Clarification_question": 6, "Acknowledgement": 7, "Conditional": 8, "Alternation": 9, "Q-Elab": 10, "Parallel": 11,
    "Continuation": 12, "Narration": 13, "Comment": 14, "Result": 15
}

MAP_SPEAKERS = {
    "ID": 0, "UI": 1, "ME": 2, "PM": 3
}


def seeding_split(seed):
    random.seed(seed)


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
                            "speaker": MAP_SPEAKERS[u["speaker"]],
                            "text": u["text"],
                            "label": -1,
                            "adj_list": []
                        }
    for root, _, file_list in os.walk(path):
        for f in file_list:
            dialogue_id = f.split(".")[0]
            if f[-3:] == "txt":
                with open(os.path.join(root, f), "r") as tf:
                    for line in tf.readlines():
                        u1, r, u2 = line[:-1].split(" ")
                        uid1 = dialogue_id + "_" + u1
                        uid2 = dialogue_id + "_" + u2
                        utterances[uid1]["adj_list"].append((uid2, MAP_RELATIONS[r]))
                        utterances[uid2]["adj_list"].append((uid1, MAP_RELATIONS[r]))
    return utterances


def collate_fn(data):
    edges = []
    edge_types = []
    uid = [d["id"] for d in data]
    uid2idx = {}
    for p in list(zip(uid, np.arange(len(uid)))):
        uid2idx[p[0]] = p[1]
    for d in data:
        for adj in d["adj_list"]:
            if adj[0] in uid:
                edges.append([uid2idx[d["id"]], uid2idx[adj[0]]])
                edge_types.append(adj[1])
    return {
        "text": [d["text"] for d in data],
        "speaker": torch.tensor([d["speaker"] for d in data]),
        "label": torch.tensor([d["label"] for d in data]),
        "edge_index": torch.tensor(edges).transpose(0, 1).type(torch.int64),
        "edge_type": torch.tensor(edge_types)
    }


class CompleteDataset(Dataset):
    def __init__(self,
                 split,
                 config):
        super(CompleteDataset, self).__init__()

        assert split in ["train", "dev", "test"]

        self.data = []

        if split in ["train", "dev"]:
            # read training dialogues
            utterances = read_dialogues(os.path.join(config.Path.data, "training"))
            # read training labels
            with open(os.path.join(config.Path.data, "training_labels.json")) as lf:
                training_labels = json.load(lf)
                for dialogue_id, labels in training_labels.items():
                    for idx, label in enumerate(labels):
                        utterance_id = dialogue_id + "_" + str(idx)
                        utterances[utterance_id]["label"] = label
            # split training and dev set
            self.data = list(utterances.values())
            dev_train_ratio = config.DownStream.dev_train_ratio
            dev_cnt = int(len(self.data) * dev_train_ratio)
            train_cnt = len(self.data) - dev_cnt
            if split == "train":
                self.data = self.data[:train_cnt]
            else:
                self.data = self.data[-dev_cnt:]
        elif split == "test":
            # read testing dialogues
            utterances = read_dialogues(os.path.join(config.Path.data, "test"))
            self.data = list(utterances.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def complete_dataloader(subset, config, batch_size):
    dataset = CompleteDataset(subset, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
