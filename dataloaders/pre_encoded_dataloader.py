import json
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import torch
import os
import random
from queue import Queue
import pickle
import configs.pre_encoded_gcn_config as base_config

MAP_RELATIONS = {
    "Elaboration": 0, "Question-answer_pair": 1, "Correction": 2, "Contrast": 3, "Background": 4, "Explanation": 5,
    "Clarification_question": 6, "Acknowledgement": 7, "Conditional": 8, "Alternation": 9, "Q-Elab": 10, "Parallel": 11,
    "Continuation": 12, "Narration": 13, "Comment": 14, "Result": 15
}

MAP_SPEAKERS = {
    "ID": 0, "UI": 1, "ME": 2, "PM": 3
}

MAP_SPEAKER_TURN = {}

for spk_1 in MAP_SPEAKERS.values():
    for spk_2 in MAP_SPEAKERS.values():
        MAP_SPEAKER_TURN[str(spk_1) + "_" + str(spk_2)] = len(MAP_SPEAKER_TURN)


def graph_search(n, mat, size):
    output = []
    q = Queue()
    q.put(n)
    while len(output) < size and q.qsize() > 0:
        node = q.get()
        output.append(node)
        for idx, state in enumerate(mat[node, :]):
            if state == 1 and idx not in output:
                q.put(idx)
    return output


def read_dialogues(dialogue_path, encoding_path, config):
    utterances = {}
    dialogues = {}

    encoding_file = open(encoding_path, "rb")
    encoding_data = pickle.load(encoding_file)

    for root, _, file_list in os.walk(dialogue_path):
        for f in file_list:
            dialogue_id = f.split(".")[0]
            if f[-4:] == "json":
                dialogues[dialogue_id] = {
                    "size": 0
                }
                with open(os.path.join(root, f), "r") as jf:
                    us = json.load(jf)
                    for u in us:
                        utterance_id = dialogue_id + "_" + str(u["index"])
                        utterances[utterance_id] = {
                            "id": utterance_id,
                            "speaker": MAP_SPEAKERS[u["speaker"]],
                            "text": u["text"],
                            "label": -1,
                            "adj_list": [],
                            "encoding": torch.zeros([config.Model.bert_embsize], dtype=torch.float64)
                        }
                        dialogues[dialogue_id]["size"] += 1
    for _, dialogue in dialogues.items():
        dialogue["adj_matrix"] = np.zeros((dialogue["size"], dialogue["size"])).astype(int)
    for root, _, file_list in os.walk(dialogue_path):
        for f in file_list:
            dialogue_id = f.split(".")[0]
            if f[-3:] == "txt":
                with open(os.path.join(root, f), "r") as tf:
                    for line in tf.readlines():
                        u1, r, u2 = line[:-1].split(" ")
                        uid1 = dialogue_id + "_" + u1
                        uid2 = dialogue_id + "_" + u2
                        if config.Model.speaker_relation:
                            spk_1 = utterances[uid1]["speaker"]
                            spk_2 = utterances[uid2]["speaker"]
                            speaker_relation = MAP_SPEAKER_TURN[str(spk_1) + "_" + str(spk_2)]
                            utterances[uid1]["adj_list"].append((uid2, MAP_RELATIONS[r], speaker_relation))
                        else:
                            utterances[uid1]["adj_list"].append((uid2, MAP_RELATIONS[r]))
                        dialogues[dialogue_id]["adj_matrix"][int(u1), int(u2)] = 1
                        dialogues[dialogue_id]["adj_matrix"][int(u2), int(u1)] = 1

    for i in range(len(encoding_data["id"])):
        utterances[encoding_data["id"][i]]["encoding"] = encoding_data["embeddings"][i]

    return utterances, dialogues


def collate_fn(data):
    edges = []
    edge_types = []
    edge_speaker_types = []
    uid = [d["id"] for d in data]
    uid2idx = {}
    encoding = []
    for p in list(zip(uid, np.arange(len(uid)))):
        uid2idx[p[0]] = p[1]
    for d in data:
        encoding.append(d["encoding"].tolist())
        for adj in d["adj_list"]:
            if adj[0] in uid:
                edges.append([uid2idx[d["id"]], uid2idx[adj[0]]])
                edge_types.append(adj[1])
                if base_config.Model.speaker_relation:
                    edge_speaker_types.append(adj[2])
    if base_config.Model.speaker_relation:
        return {
            "id": [d["id"] for d in data],
            "text": [d["text"] for d in data],
            "speaker": torch.tensor([d["speaker"] for d in data]),
            "label": torch.tensor([d["label"] for d in data]),
            "edge_index": torch.tensor(edges).transpose(0, 1).type(torch.int64),
            "edge_type": torch.tensor(edge_types),
            "edge_speaker_type": torch.tensor(edge_speaker_types),
            "encoding": torch.tensor(encoding)
        }
    else:
        return {
            "id": [d["id"] for d in data],
            "text": [d["text"] for d in data],
            "speaker": torch.tensor([d["speaker"] for d in data]),
            "label": torch.tensor([d["label"] for d in data]),
            "edge_index": torch.tensor(edges).transpose(0, 1).type(torch.int64),
            "edge_type": torch.tensor(edge_types),
            "encoding": torch.tensor(encoding)
        }


class PreEncodedDataset(Dataset):
    def __init__(self,
                 split,
                 config,
                 dev_idx=None):
        super(PreEncodedDataset, self).__init__()

        assert split in ["train", "dev", "full", "test"]

        self.data = []
        self.all_idx = []
        self.positive_idx = []
        self.negative_idx = []
        self.uid2idx = {}

        if split in ["train", "dev", "full"]:
            # read training dialogues
            utterances, self.dialogues = read_dialogues(os.path.join(config.Path.data, "training"),
                                                        os.path.join(config.Path.data, "encoding/train-dev.pkl"),
                                                        config)
            # read training labels
            with open(os.path.join(config.Path.data, "training_labels.json")) as lf:
                training_labels = json.load(lf)
                for dialogue_id, labels in training_labels.items():
                    for idx, label in enumerate(labels):
                        utterance_id = dialogue_id + "_" + str(idx)
                        utterances[utterance_id]["label"] = label
            # split training and dev set
            self.data = list(utterances.values())
            for idx, utt in enumerate(self.data):
                self.uid2idx[utt["id"]] = idx
            dev_train_ratio = config.DownStream.dev_train_ratio

            step_size = int(len(self.data) * dev_train_ratio)

            dev_start = dev_idx * step_size
            dev_end = (dev_idx + 1) * step_size

            if split == "train":
                for idx, utt in enumerate(self.data):
                    if idx < dev_start or idx >= dev_end:
                        self.all_idx.append(idx)
                        if utt["label"] == 1:
                            self.positive_idx.append((idx, utt["id"]))
                        else:
                            self.negative_idx.append((idx, utt["id"]))
            elif split == "dev":
                self.data = self.data[dev_start:dev_end]
            else:
                for idx, utt in enumerate(self.data):
                    self.all_idx.append(idx)
                    if utt["label"] == 1:
                        self.positive_idx.append((idx, utt["id"]))
                    else:
                        self.negative_idx.append((idx, utt["id"]))
        elif split == "test":
            # read testing dialogues
            utterances, self.dialogues = read_dialogues(os.path.join(config.Path.data, "test"),
                                                        os.path.join(config.Path.data, "encoding/test.pkl"),
                                                        config)
            self.data = list(utterances.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, list):
            # Handling a batch of indices
            batch = [self.data[i] for i in index if i != -1]
            return batch
        else:
            # Handling a single index
            return self.data[index]

    def pos_indices(self):
        return self.positive_idx

    def neg_indices(self):
        return self.negative_idx

    def adj_mat(self):
        return self.dialogues

    def adj_matrix(self):
        return self.dialogues

    def uid_map(self):
        return self.uid2idx

    def idx_list(self):
        return self.all_idx


class CustomSampler(Sampler):
    def __init__(self,
                 all_idx,
                 dialogues,
                 uid_map,
                 pos_indices,
                 neg_indices,
                 batch_size):
        # super().__init__()
        self.idx_list = all_idx
        self.dataset_len = len(all_idx)
        self.dialogues = dialogues
        self.uid_map = uid_map
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size

    def __iter__(self):
        # sampling roots for width-first search
        # half positive roots and half negative roots
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)
        root_cnt = self.dataset_len // self.batch_size
        roots = []
        roots.extend(self.pos_indices[:root_cnt // 2])
        roots.extend(self.neg_indices[:root_cnt - root_cnt // 2])
        random.shuffle(roots)
        batch = []
        for idx, utt_id in roots:
            dialogue_id, n = utt_id.split("_")
            mat = self.dialogues[dialogue_id]["adj_matrix"]
            n = int(n)
            indices = graph_search(n, mat, self.batch_size)
            for i in indices:
                index = self.uid_map[dialogue_id + "_" + str(i)]
                # prevent information leak between train and dev split
                if index in self.idx_list:
                    batch.append(index)

            if len(batch) >= self.batch_size or idx == self.pos_indices[-1]:
                if len(batch) < self.batch_size:
                    batch.extend([-1] * (self.batch_size - len(batch)))

                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

    def __len__(self):
        return self.dataset_len // self.batch_size


def pre_encoded_dataloader(subset, config, batch_size, dev_idx=None, custom_sampler=False):
    dataset = PreEncodedDataset(subset, config, dev_idx)
    if subset == "train" and custom_sampler:
        sampler = CustomSampler(all_idx=dataset.idx_list(),
                                dialogues=dataset.adj_mat(),
                                uid_map=dataset.uid_map(),
                                pos_indices=dataset.pos_indices(),
                                neg_indices=dataset.neg_indices(),
                                batch_size=batch_size)
        return DataLoader(dataset, batch_size=None, sampler=sampler, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
