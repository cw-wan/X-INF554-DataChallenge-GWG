import os
import torch
from tqdm import tqdm
import transformers
from utils.common import write_log
from modules import PreEncodedGCN
from configs import pre_encoded_gcn_config
from dataloaders.pre_encoded_dataloader import pre_encoded_dataloader
from sklearn.metrics import accuracy_score, f1_score
from utils.training_utils import f1_score_macro, seed_everything


def eval_pre_encoded_gcn(model, config, dev_idx):
    with torch.no_grad():
        model.eval()
        eval_dataset = pre_encoded_dataloader(subset="dev", config=config, batch_size=config.DownStream.batch_size,
                                              dev_idx=dev_idx)
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


def train_pre_encoded_gcn(config=pre_encoded_gcn_config):
    # set seed
    seed_everything(config.DownStream.seed)
    device = config.device
    # load training parameters
    learning_rate = config.DownStream.learning_rate
    batch_size = config.DownStream.batch_size
    warm_up = config.DownStream.warm_up
    total_epoch = config.DownStream.total_epoch
    decay = config.DownStream.decay

    dev_f1_macro_score = []
    final_f1_macro_score = 0
    total_num_fold = int(round(1.0 / config.DownStream.dev_train_ratio))
    # for dev_idx in range(total_num_fold):
    for dev_idx in range(1):
        # init model
        model = PreEncodedGCN()
        model.to(device)
        # init dataloader
        train_dataloader = pre_encoded_dataloader(subset="train", config=config, batch_size=batch_size,
                                                  dev_idx=dev_idx, custom_sampler=config.DownStream.custom_sampler)

        # weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in model.named_parameters() if
                           p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           p.requires_grad and any(nd in n for nd in no_decay)],
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
        acc, f1, f1_macro = eval_pre_encoded_gcn(model, config, dev_idx=dev_idx)
        current_dev_f1_macro = 0
        print("Before training, Accuracy {}, F1 Score {}, F1 Macro Score {}, Dev_set Index {}/{}".format(acc, f1,
                                                                                                         f1_macro,
                                                                                                         dev_idx + 1,
                                                                                                         total_num_fold))
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
            acc, f1, f1_macro = eval_pre_encoded_gcn(model, config, dev_idx=dev_idx)
            current_dev_f1_macro = max(current_dev_f1_macro, f1_macro)
            log = "Epoch {}, Accuracy {}, F1 Score {}, F1 Macro Score {}, Dev_set Index {}/{}".format(epoch, acc, f1,
                                                                                                      f1_macro,
                                                                                                      dev_idx + 1,
                                                                                                      total_num_fold)
            print(log)
            write_log(log, path=model.name + '_train.log')
        dev_f1_macro_score.append(current_dev_f1_macro)
        final_f1_macro_score = final_f1_macro_score + current_dev_f1_macro
        log = "Dev_set Index {}/{}, Highest F1: {}".format(dev_idx + 1, total_num_fold, current_dev_f1_macro)
        print(log)
        write_log(log, path=model.name + '_train.log')
    final_f1_macro_score = final_f1_macro_score / total_num_fold
    log = "Total Epoch {}, Learning Rate {}, F1 {}".format(total_epoch, learning_rate, final_f1_macro_score)
    print(log)
    write_log(log, path=model.name + '_train.log')


def test_pre_encoded_gcn(load_epoch, config=pre_encoded_gcn_config):
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
    model = PreEncodedGCN()
    model.to(device)
    # init dataloader
    train_dataloader = pre_encoded_dataloader(subset="full", config=config, batch_size=batch_size,
                                              dev_idx=0, custom_sampler=config.DownStream.custom_sampler)

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

    with torch.no_grad():
        model.eval()
        test_dataloader = pre_encoded_dataloader(subset="test", config=config, batch_size=config.DownStream.batch_size)
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
        file = open("output/submission_gcn_roberta_epoch" + ".csv", "w")
        file.write("id,target_feature\n")
        for row in zip(utt_ids, predictions):
            file.write(",".join(row))
            file.write("\n")
        file.close()
