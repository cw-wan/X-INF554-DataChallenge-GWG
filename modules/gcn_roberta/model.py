from torch import nn
import torch
import os
from torch_geometric.nn.conv import RGCNConv, GraphConv
from transformers import AutoModel
from transformers import RobertaTokenizer
from modules.decoder import BaseClassifier
from configs import gcn_roberta_config as base_config


class GCNRoBERTa(nn.Module):
    def __init__(self,
                 config=base_config):
        super(GCNRoBERTa, self).__init__()

        self.config = config
        self.device = self.config.device
        self.name = "gcn_roberta"

        # RoBERTa Model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.Path.roberta, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(self.config.Path.roberta, local_files_only=True)

        # Speaker embedding layer
        if self.config.Model.speaker_embedding:
            self.spk_emb = nn.Embedding(
                num_embeddings=self.config.Model.n_speakers,
                embedding_dim=self.config.Model.roberta_embedding_size)

        # GRU
        if self.config.Model.gru:
            self.gru = nn.GRU(
                input_size=self.config.Model.roberta_embedding_size,
                hidden_size=self.config.Model.gru_hidden_size,
                num_layers=self.config.Model.gru_layers,
                bidirectional=self.config.Model.bidirectional_gru,
                dropout=self.config.Model.dropout)

        self.embedding_size = self.config.Model.gru_hidden_size * 2 if self.config.Model.gru and self.config.Model.bidirectional_gru else self.config.Model.gru_hidden_size

        # GCN
        if self.config.Model.gcn:
            self.conv1 = RGCNConv(
                in_channels=self.embedding_size,
                out_channels=self.embedding_size,
                num_relations=self.config.Model.n_relations,
                num_bases=self.config.Model.rgcn_reg_basis)
        self.conv2 = GraphConv(
            in_channels=self.embedding_size,
            out_channels=self.embedding_size)

        # MLP Classifier
        hidden_size = [int(self.embedding_size / 4), ]
        self.decoder = BaseClassifier(
            input_size=self.embedding_size,
            hidden_size=hidden_size,
            output_size=self.config.DownStream.output_size)

        # Loss
        self.criterion = nn.BCELoss()

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings[:, 0]

    def forward(self, sample, return_loss=True):
        text = sample["text"]
        spk = sample["speaker"].to(self.device)
        # transform edges to edge_index type
        edge_index = sample["edge_index"].to(self.device)
        edge_type = sample["edge_type"].to(self.device)
        # take [CLS] of RoBERTa output as the embedding of each utterance
        utt_emb = self.encode(text)
        # incorporating speaker information with speaker embedding
        if self.config.Model.speaker_embedding:
            speaker_emb = self.spk_emb(spk)
            utt_emb = utt_emb + speaker_emb
        # using GRU to obtain context information
        if self.config.Model.gru:
            utt_emb, _ = self.gru(utt_emb)
        # compute graph embeddings
        if self.config.Model.gcn:
            utt_emb = self.conv1(x=utt_emb, edge_index=edge_index, edge_type=edge_type)
            utt_emb = self.conv2(x=utt_emb, edge_index=edge_index)
        # make classification based on utterance embeddings
        pred = self.decoder(utt_emb).squeeze(1)
        if return_loss:
            labels = sample["label"].to(self.device)
            loss = self.criterion(pred.float(), labels.float())
            return loss, pred
        else:
            return pred

    def save_model(self, epoch):
        save_path = os.path.join(self.config.Path.save, self.name + str(epoch) + ".pth")
        print(self.name + " saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save, self.name + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))
