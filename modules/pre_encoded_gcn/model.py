from torch import nn
import torch
import os
from torch_geometric.nn.conv import RGCNConv, GraphConv
from modules.decoder import BaseClassifier
from configs import pre_encoded_gcn_config as base_config
from modules.loss import f1_loss
from pytorch_metric_learning.losses import NTXentLoss


class PreEncodedGCN(nn.Module):
    def __init__(self,
                 config=base_config):
        super(PreEncodedGCN, self).__init__()

        self.config = config
        self.device = self.config.device
        self.name = "pre_encoded_gcn"

        # Speaker embedding layer
        if self.config.Model.speaker_embedding:
            self.spk_emb = nn.Embedding(
                num_embeddings=self.config.Model.n_speakers,
                embedding_dim=self.config.Model.speaker_embedding_size)

        # GRU - OPTIONAL
        if self.config.Model.gru:
            self.gru = nn.GRU(
                input_size=self.config.Model.bert_embsize,
                hidden_size=self.config.Model.bert_embsize // 2 if self.config.Model.gru_bidirect else self.config.Model.bert_embsize,
                num_layers=self.config.Model.gru_layers,
                bidirectional=self.config.Model.gru_bidirect,
                dropout=self.config.Model.dropout)

        self.embedding_size = self.config.Model.utt_size

        # Projection layer for dim reduction - OPTIONAL
        if self.config.Model.shrink_before_gcn:
            shrink_list = []
            shrink_list.append(nn.Linear(self.config.Model.utt_size, self.config.Model.shrink_output_size))
            shrink_list.append(nn.GELU())
            self.shrink = nn.Sequential(*shrink_list)
            self.embedding_size = self.config.Model.shrink_output_size

        # GCN, relation types depend on the graph type: utterance dependency or speaker turn
        n_relations = 0
        if config.Model.graph_type == 1:
            n_relations = 16
        elif config.Model.graph_type == 2:
            n_relations = 32
        if config.Model.speaker_relation:
            n_relations = n_relations * config.Model.n_speakers * config.Model.n_speakers
        if self.config.Model.gcn:
            self.conv1 = RGCNConv(
                in_channels=self.embedding_size,
                out_channels=self.embedding_size,
                num_relations=n_relations,
                num_bases=self.config.Model.rgcn_reg_basis)
            self.conv2 = GraphConv(
                in_channels=self.embedding_size,
                out_channels=self.embedding_size)

        # MLP Classifier
        hidden_size = [int(self.embedding_size / 2), int(self.embedding_size / 4), ]
        self.decoder = BaseClassifier(
            input_size=self.embedding_size * 2,
            hidden_size=hidden_size,
            output_size=self.config.DownStream.output_size)

        # Loss
        self.criterion = f1_loss
        self.info_nce = NTXentLoss(temperature=self.config.Model.temperature)

    def forward(self, sample, return_loss=True):
        text = sample["text"]
        spk = sample["speaker"].to(self.device)
        edge_index = None
        edge_type = None
        if self.config.Model.graph_type == 1:
            edge_index = sample["edge_index"].to(self.device)
            edge_type = sample["edge_type"].to(self.device)
        elif self.config.Model.graph_type == 2:
            edge_index = sample["edge_index_2"].to(self.device)
            edge_type = sample["edge_type_2"].to(self.device)
        # take [CLS] of RoBERTa output as the embedding of each utterance
        utt_emb = sample["encoding"].to(self.device)
        # incorporating speaker information with speaker embedding
        if self.config.Model.speaker_embedding:
            speaker_emb = self.spk_emb(spk)
            utt_emb = torch.cat((utt_emb, speaker_emb), 1)
        if self.config.Model.shrink_before_gcn:
            utt_emb = self.shrink(utt_emb)
        # using GRU to obtain context information
        if self.config.Model.gru:
            utt_emb, _ = self.gru(utt_emb)
        # compute graph embeddings
        if self.config.Model.gcn:
            utt_emb_before_gcn = utt_emb
            utt_emb = self.conv1(utt_emb, edge_index, edge_type)
            utt_emb = self.conv2(x=utt_emb, edge_index=edge_index)
            utt_emb = torch.cat((utt_emb_before_gcn, utt_emb), 1)
        # make classification based on utterance embeddings
        pred = self.decoder(utt_emb).squeeze(1)
        if return_loss:
            labels = sample["label"].to(self.device)
            pred_loss = self.criterion(pred.float(), labels.float())
            contrastive_loss = self.info_nce(embeddings=utt_emb, labels=labels)
            loss = pred_loss + contrastive_loss * 0.5
            return loss, pred_loss, contrastive_loss, pred
        else:
            return pred

    def save_model(self, epoch):
        if not os.path.exists(self.config.Path.save):
            os.mkdir(self.config.Path.save)
        save_path = os.path.join(self.config.Path.save, self.name + str(epoch) + ".pth")
        print(self.name + " saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save, self.name + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))
