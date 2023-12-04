from dataloaders.tuner_dataloader import tuner_dataloader
from configs import roberta_tuner_config
from tqdm import tqdm
import os

os.chdir("../")

test_dataloader = tuner_dataloader(split="test", config=roberta_tuner_config, shuffle=False, batch_size=64)

bar = tqdm(test_dataloader)

for index, sample in enumerate(bar):
    print(sample["id"])
