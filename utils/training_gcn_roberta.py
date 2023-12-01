import os
import torch
from tqdm import tqdm
import transformers
from utils.common import write_log
from modules import GCNRoBERTa
from configs import gcn_roberta_config
from dataloaders.complete_dataloader import complete_dataloader
from sklearn.metrics import accuracy_score, f1_score
from utils.training_utils import f1_score_macro, seed_everything


def eval_gcn_roberta(model):
    with torch.no_grad():
        model.eval()
        eval_dataset = complete_dataloader(subset="dev", config=gcn_roberta_config, batch_size=32)
        float_pred = []
        predictions = []
        truths = []
        bar = tqdm(eval_dataset)
        for index, sample in enumerate(bar):
            truths.append(sample["label"])
            pred = model(sample, return_loss=False)
            float_pred.append(pred)
            predictions.append(torch.round(pred))
        f1_macro = f1_score_macro(torch.cat(float_pred, dim=-1).cpu().detach(),
                                  torch.cat(truths, dim=-1).cpu().detach())
        predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy()
        truths = torch.cat(truths, dim=-1).detach().numpy()
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions)
    return acc, f1, f1_macro


def train_gcn_roberta(config=gcn_roberta_config):
    # set seed
    seed_everything(config.DownStream.seed)
    device = config.device
    # load training parameters
    learning_rate = config.DownStream.learning_rate
    batch_size = config.DownStream.batch_size
    warm_up = config.DownStream.warm_up
    total_epoch = config.DownStream.total_epoch
    decay = config.DownStream.decay

    # init model
    model = GCNRoBERTa()
    model.to(device)

    # init dataloader
    train_dataloader = complete_dataloader(subset="train", config=config, batch_size=batch_size)

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # init optimizer
    optimizer = torch.optim.AdamW(params=optimizer_grouped_params, lr=learning_rate, amsgrad=False)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )

    # train
    pred_loss = torch.tensor(0)
    contrastive_loss = torch.tensor(0)
    loss = torch.tensor(0)
    acc, f1, f1_macro = eval_gcn_roberta(model)
    print("Before training, Accuracy {}, F1 Score {}, F1 Macro Score {}".format(acc, f1, f1_macro))
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:%s|PredLoss:%s|ContLoss:%s" % (
                epoch, loss.item(), pred_loss.item(), contrastive_loss.item()))
            loss, pred_loss, contrastive_loss, pred = model(sample)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # evaluate
        acc, f1, f1_macro = eval_gcn_roberta(model)
        log = "Epoch {}, Accuracy {}, F1 Score {}, F1 Macro Score {}".format(epoch, acc, f1, f1_macro)
        print(log)
        write_log(log, path='gcn_roberta_train.log')
        # save model
        model.save_model(epoch)


def test_gcn_roberta(load_epoch, config=gcn_roberta_config):
    device = config.device
    # load trained model
    model = GCNRoBERTa()
    model.load_model(load_epoch)
    model.to(device)
    with torch.no_grad():
        model.eval()
        test_dataloader = complete_dataloader(subset="test", config=config, batch_size=32)
        predictions = []
        utt_ids = []
        bar = tqdm(test_dataloader)
        for index, sample in enumerate(bar):
            pred = model(sample, return_loss=False)
            utt_ids += sample["id"]
            predictions.append(torch.round(pred))
        predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy().tolist()
        predictions = [str(p) for p in predictions]
        # write to submission file
        if not os.path.exists("output/"):
            os.mkdir("output/")
        file = open("output/submission_gcn_roberta_epoch" + str(load_epoch) + ".csv", "w")
        file.write("id,target_feature\n")
        for row in zip(utt_ids, predictions):
            file.write(",".join(row))
            file.write("\n")
        file.close()
