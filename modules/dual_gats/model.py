from torch import nn
import torch
import os
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from transformers import AutoModel, RobertaTokenizer
from modules.decoder import BaseClassifier
from modules.loss import f1_loss, diff_loss
from modules.gan import GATBlock
from configs import dual_gats_config as base_config


class DualGATs(nn.Module):
    def __init__(self,
                 config=base_config):
        super(DualGATs, self).__init__()

        self.config = config
        self.device = self.config.device
        self.name = "dual_GATs"

        # RoBERTa Model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.Path.roberta, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(self.config.Path.roberta, local_files_only=True)

        self.embedding_size = self.config.Model.bert_embsize

        # Dual Graph Attention Networks
        self.edge_fea_emb_1 = nn.Embedding(16, self.config.Model.edge_fea)
        self.edge_fea_emb_2 = nn.Embedding(32, self.config.Model.edge_fea)
        self.gan_discourse = []
        self.gan_speaker = []
        for i in range(self.config.Model.gan_layers):
            self.gan_discourse.append(
                GATBlock(self.embedding_size, self.config.Model.ll_hidden_size, self.config.Model.gat_heads,
                         self.config.Model.edge_fea).to(self.device))
            self.gan_speaker.append(
                GATBlock(self.embedding_size, self.config.Model.ll_hidden_size, self.config.Model.gat_heads,
                         self.config.Model.edge_fea).to(self.device))

        # Cross-Attention Matrix
        self.affine1 = nn.Parameter(torch.empty(size=(self.embedding_size, self.embedding_size)))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=(self.embedding_size, self.embedding_size)))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        # MLP Classifier
        hidden_size = [int(self.embedding_size / 4), ]
        self.decoder = BaseClassifier(
            input_size=self.embedding_size * 2,
            hidden_size=hidden_size,
            output_size=self.config.DownStream.output_size)

        # Loss
        self.criterion = f1_loss
        self.diff_loss = diff_loss
        self.info_nce = NTXentLoss(temperature=self.config.Model.temperature)

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings[:, 0]

    def forward(self, sample, return_loss=True):
        text = sample["text"]
        edge_index_1 = sample["edge_index"].to(self.device)
        edge_type_1 = sample["edge_type"].to(self.device)
        edge_index_2 = sample["edge_index_2"].to(self.device)
        edge_type_2 = sample["edge_type_2"].to(self.device)
        edge_fea_1 = self.edge_fea_emb_1(edge_type_1)
        edge_fea_2 = self.edge_fea_emb_2(edge_type_2)
        # take [CLS] of RoBERTa output as the embedding of each utterance
        utt_emb = self.encode(text)
        differential_loss = 0
        H_discourse = [utt_emb, ]
        H_speaker = [utt_emb, ]
        for i in range(self.config.Model.gan_layers):
            h_dis, _ = self.gan_discourse[i](H_discourse[i], edge_index_1, edge_fea_1)
            h_spk, _ = self.gan_speaker[i](H_speaker[i], edge_index_2, edge_fea_2)

            differential_loss = differential_loss + self.diff_loss(h_dis, h_spk)

            # Cross Attention (problematic)
            """
            A1 = F.softmax(h_dis @ self.affine1 @ h_spk.transpose(0, 1), dim=-1)
            A2 = F.softmax(h_spk @ self.affine2 @ h_dis.transpose(0, 1), dim=-1)
            h_dis = A1 @ h_spk
            h_spk = A2 @ h_dis
            """
            H_discourse.append(h_dis)
            H_speaker.append(h_spk)
        differential_loss = differential_loss / self.config.Model.gan_layers
        utt_emb = torch.cat((H_discourse[-1], H_speaker[-1]), dim=-1)
        pred = self.decoder(utt_emb).squeeze(1)
        if return_loss:
            labels = sample["label"].to(self.device)
            pred_loss = self.criterion(pred.float(), labels.float())
            contrastive_loss = self.info_nce(embeddings=utt_emb, labels=labels)
            loss = pred_loss + contrastive_loss * self.config.Model.lambda_contrast + differential_loss * self.config.Model.lambda_diff
            return loss, pred_loss, contrastive_loss, differential_loss, pred
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

    def freeze_roberta(self):
        for n, param in self.encoder.named_parameters():
            param.requires_grad = False
