import os
import random

import torch
import easydict
import numpy as np

from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.criterion import get_criterion


def get_args():
    config = {}

    # 설정
    config["seed"] = 42
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터
    config["max_seq_len"] = 20
    config["num_workers"] = 1
    config["data_dir"] = "/opt/ml/input/data/train_dataset"

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


def get_root_dir(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    return root_dir


def set_seeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
