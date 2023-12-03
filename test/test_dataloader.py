from dataloaders.complete_dataloader import complete_dataloader, MAP_SPEAKER_TURN
from configs import gcn_roberta_config
from tqdm import tqdm
import os

print(MAP_SPEAKER_TURN)

os.chdir("../")

dev = complete_dataloader(subset="dev", config=gcn_roberta_config, batch_size=64)

bar = tqdm(dev)

positive_utts = []
negative_utts = []

for i, sample in enumerate(bar):
    pids = []
    for _id, l in enumerate(sample["label"]):
        if l == 1:
            pids.append(_id)
    for _id, t in enumerate(sample["text"]):
        if _id in pids:
            positive_utts.append(t)
        else:
            negative_utts.append(t)

with open("positive_samples.txt", 'a') as f:
    f.writelines("\n".join(positive_utts))
with open("negative_samples.txt", 'a') as f:
    f.writelines("\n".join(negative_utts))
"""
bar = tqdm(train_normal)

positive = 0
allcnt = 0

for i, sample in enumerate(bar):
    allcnt += len(sample["id"])
    positive += sum(sample["label"])

print("Proportion of positive without custom sampler: {}".format(positive / allcnt))
"""
