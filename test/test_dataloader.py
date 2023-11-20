from dataloaders.simple_dataloader import Simple
from configs import naive_roberta_config
import os

os.chdir("../")

train = Simple("train", naive_roberta_config)
dev = Simple("dev", naive_roberta_config)

train_ids = set()
dev_ids = set()

for i in range(len(train)):
    train_ids.add(train[i]["id"])

for j in range(len(dev)):
    dev_ids.add(dev[j]["id"])

print("train")
print(len(list(train_ids)))
print("dev")
print(len(list(dev_ids)))

common_ids = train_ids.intersection(dev_ids)

print("common")
print(len(list(common_ids)))
