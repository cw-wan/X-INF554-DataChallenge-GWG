import os
import torch
from torch import nn
from modules.decoder import BaseClassifier
from transformers import AutoModel
from transformers import RobertaTokenizer
from configs import sequential_roberta_config as base_config


class SequentialRoBERTa(nn.Module):
    def __init__(self,
                 config=base_config):
        super(SequentialRoBERTa, self).__init__()

        self.config = config
        self.device = self.config.device

        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.Path.roberta, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(self.config.Path.roberta, local_files_only=True)

        self.embedding_size = self.config.Model.embedding_size
        hidden_size = [int(self.embedding_size / 4), ]

        self.decoder = BaseClassifier(input_size=self.embedding_size,
                                      hidden_size=hidden_size,
                                      output_size=self.config.DownStream.output_size)

        self.criterion = nn.BCELoss()

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings

    def forward(self, samples, return_loss=True):
        cls = self.encode(samples["text"])[:, 0]
        pred = self.decoder(cls).squeeze(1)
        if return_loss:
            labels = samples["label"].to(self.device)
            loss = self.criterion(pred.float(), labels.float())
            return loss, pred
        else:
            return pred

    def save_model(self, epoch):
        save_path = os.path.join(self.config.Path.save, "sequential_roberta_" + str(epoch) + ".pth")
        print("Sequential Roberta saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save,
                                       'sequential_roberta_' + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))

