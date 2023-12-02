from dataloaders.complete_dataloader import complete_dataloader
from configs import gcn_roberta_config
from tqdm import tqdm
import os

os.chdir("../")

train = complete_dataloader(subset="train", config=gcn_roberta_config, batch_size=64, custom_sampler=True)
train_normal = complete_dataloader(subset="train", config=gcn_roberta_config, batch_size=32)

bar = tqdm(train)

positive = 0
allcnt = 0

for i, sample in enumerate(bar):
    allcnt += len(sample["id"])
    positive += sum(sample["label"])

print("Proportion of positive with custom sampler: {}".format(positive / allcnt))

bar = tqdm(train_normal)

positive = 0
allcnt = 0

for i, sample in enumerate(bar):
    allcnt += len(sample["id"])
    positive += sum(sample["label"])

print("Proportion of positive without custom sampler: {}".format(positive / allcnt))
