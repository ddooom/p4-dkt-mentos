import os
import logging
import os.path as p
from datetime import datetime

import wandb
import torch
import easydict
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from logger import get_logger
from dkt_dataset import DKTDataset
from utils.criterion import get_criterion


def get_args():
    config = {}

    # 설정
    config["seed"] = 42
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터
    config["max_seq_len"] = 20
    config["num_workers"] = 1

    # 모델
    config["hidden_dim"] = 64
    config["n_layers"] = 2
    config["dropout"] = 0.2

    # 훈련
    config["n_epochs"] = 20
    config["batch_size"] = 64
    config["lr"] = 0.0001
    config["clip_grad"] = 10
    config["log_steps"] = 50
    config["patience"] = 5

    # 중요
    config["optimizer"] = "adam"
    config["scheduler"] = "plateau"

    args = easydict.EasyDict(config)
    return args


class DKTTrainer:
    def __init__(self, args, model):
        self.args = get_args()
        self.args.update(**args)

        self.model = model
        self.root_dir = args.root_dir

        self._helper_init()

    def _helper_init(self):
        self.logger = logger
        self.prefix_save_path = datetime.now().strftime("[%m/%d_%H:%M]")
        self.prefix_save_path = f"LOG_TRAIN_{self.prefix_save_path}"

    def _update_params(self, loss, model, optimizer, args):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        optimizer.zero_grad()

    def _collate_fn(self, batch):
        col_n = len(batch[0])
        col_list = [[] for _ in range(col_n)]

        for row in batch:
            for i, col in enumerate(row):
                pre_padded = torch.zeros(self.args.max_seq_len)
                pre_padded[-len(col) :] = col
                col_list[i].append(pre_padded)

        for i, _ in enumerate(col_list):
            col_list[i] = torch.stack(col_list[i])

        return tuple(col_list)

    def _get_loaders(self, train_data, valid_data):
        trainset = DKTDataset(train_data, self.args, self.args.columns)
        validset = DKTDataset(valid_data, self.args, self.args.columns)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=self.args.num_workers,
            shuffle=True,
            batch_size=self.args.batch_size,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

        valid_loader = torch.utils.data.DataLoader(
            validset,
            num_workers=self.args.num_workers,
            shuffle=False,
            batch_size=self.args.batch_size,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

        return train_loader, valid_loader

    def _to_numpy(self, preds):
        if self.args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
        return preds

    def _save_model(self):
        save_path = p.join(self.root_dir, self.prefix_save_path)
        assert p.exists(save_path), f"{save_path} does not exist"

        # get original model if use torch.nn.DataParallel
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), f"{save_path}/model.pth")

    def _load_model(self):
        load_path = p.join(self.root_dir, self.prefix_save_path)
        model_path = f"{load_path}/model.pth"
        assert p.exists(model_path), f"{model_path} does not exist"

        # strict=False, 일치하지 않는 키들을 무시
        self.model.load_state_dict(torch.load(model_path), strict=False)

    def _get_metric(self, targets, preds):
        auc = roc_auc_score(targets, preds)
        acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))
        return auc, acc

    def _compute_loss(self, preds, targets):
        loss = get_criterion(preds, targets)

        # 마지막 Sequence에 대한 값만 Loss를 계산한다.
        loss = loss[:, -1]
        loss = torch.mean(loss)
        return loss

    def _process_batch(self, batch):
        test, question, tag, correct, mask = batch

        # change to float
        mask = mask.type(torch.FloatTensor)
        correct = correct.type(torch.FloatTensor)

        # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
        # saint의 경우 decoder에 들어가는 input이다.
        interaction = correct + 1  # for padding, 0은 padding으로 사용
        interaction = interaction.roll(shifts=1, dims=1)

        # TODO: 코드 수정되어야 합니다.
        interaction[:, 0] = 0  # set padding index to the first sequence
        interaction = (interaction * mask).to(torch.int64)

        test = ((test + 1) * mask).to(torch.int64)
        question = ((question + 1) * mask).to(torch.int64)
        tag = ((tag + 1) * mask).to(torch.int64)

        test = test.to(self.args.device)
        question = question.to(self.args.device)

        tag = tag.to(self.args.device)
        correct = correct.to(self.args.device)
        mask = mask.to(self.args.device)

        interaction = interaction.to(self.args.device)

        return (test, question, tag, correct, mask, interaction)

    def _train(self, train_loader, optimizer):
        self.model.train()

        total_preds, total_targets = [], []
        losses = []

        for step, batch in enumerate(train_loader):
            batch = self._process_batch(batch)
            preds = self.model(batch)
            targets = batch[3]  # correct

            loss = self._compute_loss(preds, targets)
            self._update_params(loss, self.model, optimizer)

            if step % self.args.log_steps == 0:
                logger.info(f"Training steps: {step} Loss: {str(loss.item())}")

            preds, targets = preds[:, -1], targets[:, -1]

            if self.args.device == "cuda":
                preds = preds.to("cpu").detach().numpy()
                targets = targets.to("cpu").detach().numpy()
            else:
                preds = preds.detach().numpy()
                targets = targets.detach().numpy()

            total_preds.append(preds)
            total_targets.append(targets)
            losses.append(loss)

        total_preds = np.concatenate(total_preds)
        total_targets = np.concatenate(total_targets)

        # Train AUC / ACC
        auc, acc = self._get_metric(total_targets, total_preds)
        loss_avg = sum(losses) / len(losses)

        return auc, acc, loss_avg

    def _validate(self, train_loader, valid_loader):
        self.model.eval()

        total_preds = []
        total_targets = []

        for step, batch in enumerate(valid_loader):
            batch = self._process_batch(batch)

            preds = self.model(batch)
            targets = batch[3]  # correct

            # predictions
            preds = preds[:, -1]
            targets = targets[:, -1]

            if self.args.device == "cuda":
                preds = preds.to("cpu").detach().numpy()
                targets = targets.to("cpu").detach().numpy()
            else:  # cpu
                preds = preds.detach().numpy()
                targets = targets.detach().numpy()

            total_preds.append(preds)
            total_targets.append(targets)

        total_preds = np.concatenate(total_preds)
        total_targets = np.concatenate(total_targets)

        # Train AUC / ACC
        auc, acc = self._get_metric(total_targets, total_preds)
        self.logger.info(f"VALID AUC : {auc} ACC : {acc}\n")

        return auc, acc, total_preds, total_targets

    def inference(self, test_data):
        self._load_model()  # loaded best model to self.model
        self.model.eval()

        _, test_loader = self._get_loaders(test_data, test_data)

        total_preds = dict()
        outputs = []

        for step, batch in enumerate(test_loader):
            batch = self._process_batch(batch)
            preds = self.model(batch)
            preds = preds[:, -1]

            preds = self._to_numpy(preds)
            total_preds += list(preds)

        # (test, question, tag, correct, mask, interaction)

    def debug(self, train_data, valid_data):
        prefix_save_path = f"LOG_DEBUG_{self.prefix_save_path}"
        logger.info(f"save_dir: {prefix_save_path}")
        self.logger.setLevel(logging.DEBUG)

        self.train()
        self.validate()

    def hyper_tune(self):
        self.logger.setLevel(logging.INFO)

        prefix_save_path = f"LOG_HYPER_{self.prefix_save_path}"
        logger.info(f"save_dir: {prefix_save_path}")

    def run(self, train_data, valid_data):
        self.logger.setLevel(logging.INFO)

        wandb.init(project="p-stage-4")
        wandb.config.update(self.args)
        wandb.watch(self.model)
        wandb.run.name = self.prefix_save_path

        train_loader, valid_loader = self.get_loaders(train_data, valid_data)

        self.args.total_steps = int(len(train_loader.dataset) / self.args.batch_size) * (self.args.n_epochs)
        self.args.warmup_steps = self.args.total_steps // 10

        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)

        best_auc = -1
        early_stopping_counter = 0

        for epoch in range(self.args.n_epochs):
            logger.info(f"Start Training: Epoch {epoch + 1}")

            train_auc, train_acc, train_loss = self._train(train_loader, self.model, optimizer)
            valid_auc, valid_acc, _, _ = self._validate(valid_loader, self.model)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_auc": train_auc,
                    "train_acc": train_acc,
                    "valid_auc": valid_auc,
                    "valid_acc": valid_acc,
                }
            )

            if valid_auc > best_auc:
                best_auc = valid_auc
                self._save_model()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.args.patience:
                    self.logger.info(f"EarlyStopping counter: {early_stopping_counter} out of {self.args.patience}")
                    break

            if self.args.scheduler == "plateau":
                scheduler.step(best_auc)
            else:
                scheduler.step()
