from dataloaders.complete_dataloader import complete_dataloader
from configs import naive_roberta_config
from tqdm import tqdm
import os
from modules import GCNRoBERTa

os.chdir("../")

train = complete_dataloader(subset="train", config=naive_roberta_config, batch_size=32)
dev = complete_dataloader(subset="dev", config=naive_roberta_config, batch_size=32)
test = complete_dataloader(subset="test", config=naive_roberta_config, batch_size=32)

bar = tqdm(dev)

model = GCNRoBERTa()
model.to(naive_roberta_config.device)
model.eval()

for i, sample in enumerate(bar):
    # print(sample["edge_index"].dtype)
    loss, pred = model(sample)
    print(loss, pred)
    break
