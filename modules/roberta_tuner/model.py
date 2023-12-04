import os
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from configs import roberta_tuner_config as base_config
from modules.decoder import BaseClassifier
from modules.loss import f1_loss


class RoBERTaTuner(nn.Module):
    def __init__(self, config=base_config):
        super(RoBERTaTuner, self).__init__()

        self.config = config
        self.device = config.device
        self.name = "roberta_tuner"

        # RoBERTa-large
        self.tokenizer = RobertaTokenizer.from_pretrained(config.Path.roberta, local_files_only=True)
        self.encoder = RobertaModel.from_pretrained(config.Path.roberta, local_files_only=True)

        # MLP classifier
        self.embsize = config.Model.roberta_embsize
        hidden_size = [int(self.embsize / 2), int(self.embsize / 4), ]
        self.decoder = BaseClassifier(
            input_size=self.embsize,
            hidden_size=hidden_size,
            output_size=config.Model.output_size)

        # Loss
        self.criterion = f1_loss

    def encode(self, texts):
        """
        Get embeddings of utterances with roberta-large.
        Use the embedding of [CLS] token as the representation for each utterance.
        :param texts: list of texts
        :return: tensor [B,D]
        """
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings[:, 0]

    def forward(self, sample, return_loss=False):
        # encode texts
        text = sample["text"]
        utt_emb = self.encode(text)
        # make classification based on utterance embeddings
        pred = self.decoder(utt_emb).squeeze(1)
        if return_loss:
            labels = sample["label"].to(self.device)
            loss = self.criterion(pred.float(), labels.float())
            return loss, pred, utt_emb
        else:
            return pred, utt_emb

    def save_model(self, epoch):
        if not os.path.exists(self.config.Path.save):
            os.mkdir(self.config.Path.save)
        save_path = os.path.join(self.config.Path.save, self.name + str(epoch) + ".pth")
        print(self.name + " saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save, self.name + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))
