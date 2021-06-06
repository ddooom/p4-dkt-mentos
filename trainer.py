import os
import json
import logging
import os.path as p
from datetime import datetime

import wandb
import torch
import numpy as np
import pandas as pd
from ray import tune
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from logger import get_logger
from dkt_dataset import DKTDataset
from utils import get_args, get_criterion, get_optimizer, get_scheduler, set_seeds


class CustomStopper(tune.Stopper):
    def __init__(self, args):
        self.should_stop = False
        self.args = args

    def __call__(self, trial_id, result):
        if not self.should_stop and result["valid_auc"] > 0.83:
            self.should_stop = True

        return self.should_stop or result["training_iteration"] >= self.args.n_epochs

    def stop_all(self):
        return self.should_stop


class DKTTrainer:
    def __init__(self, args, Model):
        self.args = get_args()
        self.args.update(**args)
        self.create_model = Model

        self._helper_init()

    def _helper_init(self):
        self.prefix_save_path = datetime.now().strftime("[%m.%d_%H:%M]")
        self.prefix_save_path = p.join(self.args.root_dir, f"LOG_{self.prefix_save_path}")

        os.mkdir(self.prefix_save_path)

    def _save_config(self, args, filename="run_config.json"):
        save_path = p.join(self.prefix_save_path, filename)

        with open(save_path, "w") as writer:
            writer.write(json.dumps(args, indent=4, ensure_ascii=False) + "\n")

    def _get_model(self):
        model = self.create_model(self.args).to(self.args.device)
        return model

    def _update_params(self, loss, model, optimizer):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
        optimizer.step()
        optimizer.zero_grad()

    def _collate_fn(self, batches):
        """ key값으로 batch 형성 """
        new_batches = {k: [] for k in batches[0].keys()}

        max_seq_len = 20

        # batch의 값들을 각 column끼리 그룹화
        for k in batches[0].keys():
            for batch in batches:
                pre_padded = torch.zeros(max_seq_len)
                pre_padded[-len(batch[k]) :] = batch[k]
                new_batches[k].append(pre_padded)

        for k in batches[0].keys():
            new_batches[k] = torch.stack(new_batches[k])

        return new_batches

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

    def _save_model(self, model, prefix=None):
        save_path = p.join(self.args.root_dir, self.prefix_save_path)
        assert p.exists(save_path), f"{save_path} does not exist"

        # get original model if use torch.nn.DataParallel
        model = model.module if hasattr(model, "module") else model
        save_path = f"{save_path}/{prefix}_model.pth" if prefix else f"{save_path}/model.pth"
        torch.save(model.state_dict(), save_path)

    def _load_model(self, prefix=None):
        load_path = p.join(self.args.root_dir, self.prefix_save_path)
        load_path = f"{load_path}/{prefix}_model.pth" if prefix else f"{load_path}/model.pth"
        assert p.exists(load_path), f"{load_path} does not exist"

        model = self._get_model()
        # strict=False, 일치하지 않는 키들을 무시
        model.load_state_dict(torch.load(load_path), strict=False)
        return model

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
        raise NotImplementedError

    def _hyper(self, args, checkpoint_dir=None):
        step = 0

        model = self._get_model()
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        if checkpoint_dir is not None:
            checkpoint_path = p.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            step = checkpoint["step"]

        while True:
            train_auc, train_acc, train_loss = self._train(model, self.train_loader, optimizer)
            valid_auc, valid_acc, _, _ = self._validate(model, self.valid_loader)

            tune.report(
                valid_auc=valid_auc,
                valid_acc=valid_acc,
                train_auc=train_auc,
                train_acc=train_acc,
                train_loss=train_loss,
            )

            step += 1

            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                checkpoint_path = p.join(checkpoint_dir, "checkpoint")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                    },
                    checkpoint_path,
                )

    def hyper(self, args, tune_args, train_data, valid_data):
        self.train_loader, self.valid_loader = self._get_loaders(train_data, valid_data)

        pbt_scheduler = tune.schedulers.PopulationBasedTraining(time_attr="training_iteration", **tune_args)

        stopper = CustomStopper(self.args)

        analysis = tune.run(
            self._hyper,
            name="pbt_lstm",
            stop=stopper,
            max_failures=3,
            num_samples=64,
            metric="",
            scheduler=pbt_scheduler,
            keep_checkpoints_num=2,
            local_dir="~/ray_results",
            checkpoint_score_attr="max-valid_auc",
            resources_per_trial={"cpu": 3, "gpu": 1},
            config=self.args,  # custom search algorithm may ignore this
        )

        return analysis

    def _train(self, model, train_loader, optimizer):
        model.train()

        total_preds, total_targets = [], []
        losses = []

        for step, batch in enumerate(train_loader):
            batch = self._process_batch(batch)
            preds = model(batch)
            targets = batch["answerCode"]  # correct

            loss = self._compute_loss(preds, targets)
            self._update_params(loss, model, optimizer)

            if step % self.args.log_steps == 0:
                print(f"Training steps: {step} Loss: {str(loss.item())}")

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

    def _validate(self, model, valid_loader):
        model.eval()

        total_preds = []
        total_targets = []

        for step, batch in enumerate(valid_loader):
            batch = self._process_batch(batch)

            preds = model(batch)
            targets = batch["answerCode"]  # correct

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
        print(f"VALID AUC : {auc} ACC : {acc}\n")

        return auc, acc, total_preds, total_targets

    def _inference(self, test_data, prefix=None):
        model = self._load_model(prefix)  # loaded best model to self.model
        model.eval()

        _, test_loader = self._get_loaders(test_data, test_data)

        total_proba_preds = []

        for step, batch in enumerate(test_loader):
            batch = self._process_batch(batch)

            fancy_index = torch.where(batch["answerCode"][:, -1] == -1)
            if fancy_index[0].size(0) == 0:
                continue

            for k in batch.keys():
                batch[k] = batch[k][fancy_index]

            preds = model(batch)
            preds = preds[:, -1]

            preds = self._to_numpy(preds)
            total_proba_preds += list(preds)

        write_path = os.path.join(self.prefix_save_path, f"{prefix}_test_results.csv")

        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for idx, proba in enumerate(total_proba_preds):
                w.write(f"{idx},{proba}\n")

    def debug(self, train_data, valid_data, test_data):
        """간단한 입,출력을 테스트합니다.
        1. Model Summary
        3. 한 개 데이터가 잘 생성되는지 체크합니다.
        4. 배치 데이터가 잘 생성되는지 체크합니다.
        5. forward를 체크합니다.
        6. Loss 계산 및, Predict를 체크합니다.
        """
        debug_file_handler = logging.FileHandler(f"{self.prefix_save_path}/debug.log")
        logger = get_logger("debug")
        logger.setLevel(logging.INFO)
        logger.addHandler(debug_file_handler)

        model = self._get_model()
        logger.info("MODEl SUMMARY\n")
        logger.info(summary(model))

        logger.info("\nCHECK DATASET")

        for dataset, name in zip([train_data, valid_data, test_data], ["TRAIN", "VALID", "TEST"]):
            logger.info(f"\n{name} EXAMPLES")
            for column, data in zip(self.args.columns, dataset[0]):
                logger.info(f"{column} : {data[:10]}")

        train_loader, valid_loader = self._get_loaders(train_data, valid_data)
        _, test_loader = self._get_loaders(test_data, test_data)

        logger.info("\nCHECK BATCH SHAPE")
        for data_loader, name in zip([train_loader, test_loader, valid_loader], ["TRAIN", "TEST", "VALID"]):
            batch = next(iter(data_loader))
            logger.info(f"\n{name} BATCH TYPE : {type(batch)}")
            logger.info(f"\n{name} BATCH LEN : {len(batch)}")
            logger.info(f"\n{name} BATCH DICT VALUE SHAPE : {batch['answerCode'].shape}")

        logger.info("\nCHECK MODEL FORWARD")

        batch = self._process_batch(batch)
        preds = model(batch)

        logger.info(f"\nPREDS SHAPE: {preds.shape}")
        logger.info(f"\nPREDS EXAMPLES: {preds[0]}")

        logger.info("\nCHECK METRICS")

        gt = batch["answerCode"]
        loss = self._compute_loss(preds, gt)

        logger.info(f"\nLOSS : {loss.item()}")

        auc, acc = self._get_metric(self._to_numpy(gt[:, -1]), self._to_numpy(preds[:, -1]))
        logger.info(f"AUC: {auc} ACC: {acc}")

    def run(self, train_data, valid_data, test_data, prefix="run"):
        self._save_config(self.args)
        set_seeds(self.args.seed)

        run_file_handler = logging.FileHandler(f"{self.prefix_save_path}/{prefix}.log")
        logger = get_logger("run")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(run_file_handler)

        model = self._get_model()
        wandb.init(project="p-stage-4", reinit=True)
        wandb.config.update(self.args)
        wandb.watch(model)
        wandb.run.name = f"{self.prefix_save_path}_{prefix}"

        train_loader, valid_loader = self._get_loaders(train_data, valid_data)

        self.args.total_steps = int(len(train_loader.dataset) / self.args.batch_size) * (self.args.n_epochs)
        self.args.warmup_steps = self.args.total_steps // 10

        if self.args.scheduler == "linear_warmup":
            self.args.scheduler_hp = {
                "num_training_steps": self.args.total_steps,
                "num_warmup_steps": self.args.warmup_steps,
            }

        optimizer = get_optimizer(model, self.args)
        scheduler = get_scheduler(optimizer, self.args)

        best_auc, best_acc = -1, -1
        early_stopping_counter = 0

        for epoch in range(self.args.n_epochs):
            logger.info(f"Start Training: Epoch {epoch + 1}")

            train_auc, train_acc, train_loss = self._train(model, train_loader, optimizer)
            valid_auc, valid_acc, _, _ = self._validate(model, valid_loader)

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

            logger.info(f"TRAIN_LOSS: {train_loss}")
            logger.info(f"TRAIN AUC: {train_auc} TRAIN ACC: {train_acc}")
            logger.info(f"VALID AUC: {valid_auc} VALID ACC: {valid_acc}\n")

            if valid_auc > best_auc:
                best_auc, best_acc = valid_auc, valid_acc
                self._save_model(model, prefix)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logger.info(f"EarlyStopping counter: {early_stopping_counter}")
                if early_stopping_counter >= self.args.patience:
                    logger.info(f"EarlyStopping counter: {early_stopping_counter} out of {self.args.patience}")
                    break

            if self.args.scheduler == "plateau":
                scheduler.step(best_auc)
            else:
                scheduler.step()

        self._inference(test_data, prefix)
        return best_auc, best_acc

    def run_cv(self, train_data, valid_data, test_data, folds: int, seeds: list):
        assert folds == len(seeds), "fold와 len(seeds)는 같은 수여야 합니다."

        total_data = np.concatenate([train_data, valid_data])
        self.args.seeds = seeds

        valid_results = {}

        for n_fold, seed in enumerate(seeds):
            self.args.seed = seed
            # TODO: User 패턴이 학습이 된다면, 충분히 데이터 유출될 수 있음
            train_data, valid_data = train_test_split(total_data, test_size=0.2, random_state=seed)
            prefix = f"cv_{n_fold}"

            best_auc, best_acc = self.run(train_data, valid_data, test_data, prefix=prefix)
            valid_results[prefix] = f"best_auc:{best_auc},best_acc:{best_acc}"

        self._save_config(valid_results, "valid_cv_results.json")

        new_df = pd.DataFrame([])

        for idx in range(folds):
            df = pd.read_csv(p.join(self.prefix_save_path, f"cv_{idx}_test_results.csv"))

            if idx == 0:
                new_df["id"] = df["id"]
                new_df["prediction"] = df["prediction"]
            else:
                new_df["prediction"] += df["prediction"]

        new_df["prediction"] /= folds
        new_df.to_csv(p.join(self.prefix_save_path, "cv_ensemble_test_results.csv"))
