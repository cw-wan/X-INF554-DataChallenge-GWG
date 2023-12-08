import os
import torch
import pickle
from tqdm import tqdm
import transformers
from utils.common import write_log
from modules import RoBERTaTuner
from configs import roberta_tuner_config
from dataloaders.tuner_dataloader import tuner_dataloader
from sklearn.metrics import accuracy_score, f1_score
from utils.training_utils import f1_score_macro, seed_everything


def eval_roberta(model, kth, config):
    with torch.no_grad():
        model.eval()
        eval_dataloader = tuner_dataloader(split="dev", config=config, batch_size=32, shuffle=False, kth=kth)
        float_pred = []
        predictions = []
        truths = []
        bar = tqdm(eval_dataloader)
        for index, sample in enumerate(bar):
            truths.append(sample["label"])
            pred, _ = model(sample, return_loss=False)
            float_pred.append(pred)
            predictions.append(torch.round(pred))
        f1_macro = f1_score_macro(torch.cat(float_pred, dim=-1).cpu().detach(),
                                  torch.cat(truths, dim=-1).cpu().detach())
        predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy()
        truths = torch.cat(truths, dim=-1).detach().numpy()
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions)
    return acc, f1, f1_macro


def train_roberta(config=roberta_tuner_config):
    # set seed
    seed_everything(config.DownStream.seed)
    device = config.device
    # load training parameters
    learning_rate = config.DownStream.learning_rate
    batch_size = config.DownStream.batch_size
    warm_up = config.DownStream.warm_up
    total_epoch = config.DownStream.total_epoch
    decay = config.DownStream.decay

    for k in range(config.DownStream.folds):
        # K-folds cross validation
        highest_f1 = 0
        best_epoch = 0

        # init model
        model = RoBERTaTuner()
        model.to(device)

        # init dataloader
        train_dataloader = tuner_dataloader(split="train", config=config, batch_size=batch_size, shuffle=True, kth=k)

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
        loss = torch.tensor(0)
        for epoch in range(1, total_epoch + 1):
            model.train()
            bar = tqdm(train_dataloader)
            for index, sample, in enumerate(bar):
                bar.set_description("Fold:%d|Epoch:%d|Loss:%s" % (k, epoch, loss.item()))
                loss, pred, _ = model(sample, return_loss=True)
                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            # evaluate
            acc, f1, f1_macro = eval_roberta(model, k, config)
            if f1 > highest_f1:
                highest_f1 = f1
                best_epoch = epoch
            log = "Epoch {}, Accuracy {}, F1 Score {}, F1 Macro Score {}".format(epoch, acc, f1, f1_macro)
            print(log)
            write_log(log, path='tuning_roberta.log')
        log = "Validation No.{}, Highest F1 Score {}, Best Epoch {}".format(k, highest_f1, best_epoch)
        print(log)
        write_log(log, path='tuning_roberta.log')


def compute_embeddings(total_epoch, config=roberta_tuner_config):
    # set seed
    seed_everything(config.DownStream.seed)
    device = config.device
    # load training parameters
    learning_rate = config.DownStream.learning_rate
    batch_size = config.DownStream.batch_size
    warm_up = config.DownStream.warm_up
    decay = config.DownStream.decay
    # load trained model
    model = RoBERTaTuner()
    model.to(device)
    # training model on train-dev set
    # init dataloader
    train_dataloader = tuner_dataloader(split="train-dev", config=config, batch_size=batch_size, shuffle=True)

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
    model.train()
    loss = torch.tensor(0)
    for epoch in range(1, total_epoch + 1):
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:%s" % (epoch, loss.item()))
            loss, pred, _ = model(sample, return_loss=True)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    # compute embeddings with trained model
    with torch.no_grad():
        model.eval()
        train_dev_embeddings = {
            "id": [],
            "embeddings": []
        }
        test_embeddings = {
            "id": [],
            "embeddings": []
        }
        # train-dev embeddings
        train_dev_dataloader = tuner_dataloader(split="train-dev", config=config, batch_size=batch_size, shuffle=False)
        bar = tqdm(train_dev_dataloader)
        for index, sample in enumerate(bar):
            train_dev_embeddings["id"].extend(sample["id"])
            train_dev_embeddings["embeddings"].append(model.encode(sample["text"]).detach().cpu())
        train_dev_embeddings["embeddings"] = torch.cat(train_dev_embeddings["embeddings"], dim=0)
        # print(len(train_dev_embeddings["id"]))
        # print(train_dev_embeddings["embeddings"].shape)
        # test embeddings and pred result
        test_dataloader = tuner_dataloader(split="test", config=config, batch_size=batch_size, shuffle=False)
        bar = tqdm(test_dataloader)
        predictions = []
        for index, sample in enumerate(bar):
            pred, embeddings = model(sample, return_loss=False)
            predictions.append(torch.round(pred))
            test_embeddings["id"].extend(sample["id"])
            test_embeddings["embeddings"].append(embeddings)
        predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy().tolist()
        predictions = [str(p) for p in predictions]
        test_embeddings["embeddings"] = torch.cat(test_embeddings["embeddings"], dim=0)
        # print(len(test_embeddings["id"]))
        # print(test_embeddings["embeddings"].shape)
        # write to submission file
        if not os.path.exists("output/"):
            os.mkdir("output/")
        file = open("output/submission_naive_roberta.csv", "w")
        file.write("id,target_feature\n")
        for row in zip(test_embeddings["id"], predictions):
            file.write(",".join(row))
            file.write("\n")
        file.close()
        # save as pkl file
        if not os.path.exists("data/encoding"):
            os.mkdir("data/encoding")
        with open("data/encoding/train-dev.pkl", 'wb') as f:
            pickle.dump(train_dev_embeddings, f)
        with open("data//encoding/test.pkl", 'wb') as f:
            pickle.dump(test_embeddings, f)
