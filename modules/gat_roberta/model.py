from torch import nn
import torch.nn.functional as F
import torch
import os
from pytorch_metric_learning.losses import NTXentLoss
from torch_geometric.nn.conv import GATv2Conv
from transformers import AutoModel, RobertaTokenizer
from modules.decoder import BaseClassifier
from modules.loss import f1_loss
from configs import gat_roberta_config as base_config


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))


class GATBlock(nn.Module):
    def __init__(self, emb_size, ll_hidden_size, att_heads, edge_fea_dim):
        super(GATBlock, self).__init__()
        self.ffn = FFN(emb_size, ll_hidden_size)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = F.dropout
        self.conv = GATv2Conv(
            in_channels=emb_size,
            out_channels=emb_size // att_heads,
            heads=att_heads,
            concat=True,
            edge_dim=edge_fea_dim
        )

    def forward(self, h, edge_index, edge_attr=None):
        u = self.ffn(h)
        v = self.layer_norm(h + self.dropout(u))
        w, att_weights = self.conv(v, edge_index, edge_attr, return_attention_weights=True)
        output = self.layer_norm(self.dropout(w) + v)
        return output, att_weights


class GATRoBERTa(nn.Module):
    def __init__(self,
                 config=base_config):
        super(GATRoBERTa, self).__init__()

        self.config = config
        self.device = self.config.device
        self.name = "gat_roberta"

        # RoBERTa Model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.Path.roberta, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(self.config.Path.roberta, local_files_only=True)

        self.embedding_size = self.config.Model.bert_embsize

        # Graph Attention Blocks
        self.edge_fea_emb = nn.Embedding(self.config.Model.n_relations, self.config.Model.edge_fea)
        self.conv1 = GATBlock(emb_size=self.embedding_size,
                              ll_hidden_size=self.config.Model.ll_hidden_size,
                              att_heads=self.config.Model.gat_heads,
                              edge_fea_dim=self.config.Model.edge_fea)
        self.conv2 = GATBlock(emb_size=self.embedding_size,
                              ll_hidden_size=self.config.Model.ll_hidden_size,
                              att_heads=self.config.Model.gat_heads,
                              edge_fea_dim=self.config.Model.edge_fea)

        # MLP Classifier
        hidden_size = [int(self.embedding_size / 4), ]
        self.decoder = BaseClassifier(
            input_size=self.embedding_size,
            hidden_size=hidden_size,
            output_size=self.config.DownStream.output_size)

        # Loss
        self.criterion = f1_loss
        self.info_nce = NTXentLoss(temperature=self.config.Model.temperature)

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings[:, 0]

    def forward(self, sample, return_loss=True):
        text = sample["text"]
        edge_index = sample["edge_index"].to(self.device)
        edge_type = sample["edge_type"].to(self.device)
        edge_fea = self.edge_fea_emb(edge_type)
        # take [CLS] of RoBERTa output as the embedding of each utterance
        utt_emb = self.encode(text)
        utt_emb, att_weights_1 = self.conv1(utt_emb, edge_index, edge_fea)
        utt_emb, att_weights_2 = self.conv2(utt_emb, edge_index, edge_fea)
        pred = self.decoder(utt_emb).squeeze(1)
        if return_loss:
            labels = sample["label"].to(self.device)
            pred_loss = self.criterion(pred.float(), labels.float())
            contrastive_loss = self.info_nce(embeddings=utt_emb, labels=labels)
            loss = pred_loss + contrastive_loss * 0.5
            return loss, pred_loss, contrastive_loss, pred, att_weights_1, att_weights_2
        else:
            return pred, att_weights_1, att_weights_2

    def save_model(self, epoch):
        if not os.path.exists(self.config.Path.save):
            os.mkdir(self.config.Path.save)
        save_path = os.path.join(self.config.Path.save, self.name + str(epoch) + ".pth")
        print(self.name + " saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save, self.name + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))
