import torch
from tqdm import tqdm
import transformers
from utils.common import write_log
from modules.sequential_roberta import SequentialRoBERTa
from configs import sequential_roberta_config
from dataloaders import simple_dataloader
from sklearn.metrics import accuracy_score, f1_score


def eval_sequential_roberta(model):
    with torch.no_grad():
        model.eval()
        eval_dataloader = simple_dataloader(subset="dev", config=sequential_roberta_config, batch_size=32,
                                            shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(eval_dataloader)
        for index, sample in enumerate(bar):
            truths.append(sample["label"])
            pred = model(sample, return_loss=False)
            predictions.append(torch.round(pred))
        predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy()
        truths = torch.cat(truths, dim=-1).cpu().detach().numpy()
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions)
    return acc, f1


def train_sequential_roberta():
    device = sequential_roberta_config.device
    # load training parameters
    batch_size = sequential_roberta_config.DownStream.batch_size
    learning_rate = sequential_roberta_config.DownStream.learning_rate
    warm_up = sequential_roberta_config.DownStream.warm_up
    total_epoch = sequential_roberta_config.DownStream.total_epoch
    decay = sequential_roberta_config.DownStream.decay

    # init model
    model = SequentialRoBERTa()
    model.to(device)

    # init dataloader
    train_dataloader = simple_dataloader(subset="train", config=sequential_roberta_config, batch_size=batch_size)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
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
    loss = 0
    acc, f1 = eval_sequential_roberta(model)
    print(acc, f1)
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:%s" % (epoch, loss))
            loss, pred = model(sample)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # evaluate
        acc, f1 = eval_sequential_roberta(model)
        log = "Test Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, f1)
        print(log)
        write_log(log, path='sequential_roberta_train.log')
        # save model
        model.save_model(epoch)


def test_sequential_roberta(load_epoch):
    device = sequential_roberta_config.device
    # load trained model
    model = SequentialRoBERTa()
    model.load_model(load_epoch)
    model.to(device)
    with torch.no_grad():
        model.eval()
        test_dataloader = simple_dataloader(subset="test", config=sequential_roberta_config, batch_size=32,
                                            shuffle=False)
        predictions = []
        utt_ids = []
        bar = tqdm(test_dataloader)
        for index, sample in enumerate(bar):
            utt_ids += sample["id"]
            pred = model(sample, return_loss=False)
            predictions.append(torch.round(pred))
        predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy().tolist()
        predictions = [str(p) for p in predictions]
        # write to submission file
        file = open("output/submission_sequential_roberta_epoch" + str(load_epoch) + ".csv", "w")
        file.write("id,target_feature\n")
        for row in zip(utt_ids, predictions):
            file.write(",".join(row))
            file.write("\n")
        file.close()


MODELS = {
    "sequential-roberta": {
        "train": train_sequential_roberta,
        "test": test_sequential_roberta
    }
}
