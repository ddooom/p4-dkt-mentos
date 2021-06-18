import os
import copy

import torch
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, Saint, TfixupBert

import wandb


# feature 추가 #1~4 수정
def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args)

    wandb.watch(model)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_auc": train_auc, 
            "train_acc":train_acc,
            "valid_auc":auc, 
            "valid_acc":acc,
            })

        ### TODO: model save or early stopping
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, f'{args.model}_{args.info}_{args.kfold}.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        inputs = process_batch(batch, args)
        preds = model(inputs)
        targets = inputs[-1] # correct


        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc} ') #LR : {optimizer.get}
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        inputs = process_batch(batch, args)

        preds = model(inputs)
        targets = inputs[-1] # correct


        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets



def inference(args, test_data):
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        inputs = process_batch(batch, args)

        preds = model(inputs)
        
        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, f"{args.model_name[:-3]}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
        
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))


def multi_inference(args, test_data):
    model_list = os.listdir(args.model_dir)
    print(model_list)

    kfold_preds = np.zeros(744)
    for model in model_list:
        args.model_name = model
        model = load_model(args)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)
        
        total_preds = []
        for step, batch in enumerate(test_loader):
            inputs = process_batch(batch, args)

            preds = model(inputs)
            
            # predictions
            preds = preds[:,-1]
            
            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()
                
            total_preds+=list(preds)
        kfold_preds += total_preds  
        
    kfold_preds /= len(model_list)

    write_path = os.path.join(args.output_dir, f"{args.model_name[:-5]}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
        
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(kfold_preds):
            w.write('{},{}\n'.format(id,p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'saint': model = Saint(args)
    if args.model == 'tfbert': model = TfixupBert(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    #1 dataloader #2와 순서를 맞춰주자
    (correct, question, test, tag, time_diff, 
    head, mid, tail, mid_tail, 
    head_answerProb, mid_answerProb, tail_answerProb,
    mask) = batch
    
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0 # set padding index to the first sequence
    interaction = (interaction * interaction_mask).to(torch.int64)
    # print(interaction)
    # exit()
    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    #2 추가 feature
    # time_median = ((time_median + 1) * mask).type(torch.FloatTensor)
    head_answerProb = ((head_answerProb + 1) * mask).type(torch.FloatTensor)
    mid_answerProb = ((mid_answerProb + 1) * mask).type(torch.FloatTensor)
    tail_answerProb = ((tail_answerProb + 1) * mask).type(torch.FloatTensor)


    time_diff = ((time_diff + 1) * mask).to(torch.int64)
    head = ((head + 1) * mask).to(torch.int64)
    mid = ((mid + 1) * mask).to(torch.int64)
    tail = ((tail + 1) * mask).to(torch.int64)
    mid_tail = ((mid_tail + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    #3 device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)

    # time_median = time_median.to(args.device)
    time_diff = time_diff.to(args.device)
    head = head.to(args.device)
    mid = mid.to(args.device)
    tail = tail.to(args.device)
    mid_tail = mid_tail.to(args.device)

    head_answerProb = head_answerProb.to(args.device)
    mid_answerProb = mid_answerProb.to(args.device)
    tail_answerProb = tail_answerProb.to(args.device)

    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    #4 
    return (test, question, tag, time_diff, head, mid, tail, mid_tail, 
            head_answerProb, mid_answerProb, tail_answerProb,
            mask, interaction, gather_index, correct)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    print("Loading Model from:", model_path, "...Finished.")
    return model


class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_data, valid_data):
        """훈련을 마친 모델을 반환한다"""

        # args update
        self.args = args

        train_loader, valid_loader = get_loaders(args, train_data, valid_data)
        
        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10
            
        model = get_model(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_model = -1
        early_stopping_counter = 0
        for epoch in tqdm(range(args.n_epochs)):

            ### TRAIN
            train_auc, train_acc, loss_avg = train(train_loader, model, optimizer, args)
            
            ### VALID
            auc, acc, preds, targets = validate(valid_loader, model, args)

            ### model save or early stopping
            if auc > best_auc:
                best_auc = auc
                best_model = copy.deepcopy(model)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                    break

            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
            else:
                scheduler.step()

        return best_model

    def evaluate(self, args, model, valid_data):
        """훈련된 모델과 validation 데이터셋을 제공하면 predict 반환"""
        pin_memory = False

        valset = DKTDataset(valid_data, args)
        valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                                   batch_size=args.batch_size,
                                                   pin_memory=pin_memory,
                                                   collate_fn=collate)

        auc, acc, preds, _ = validate(valid_loader, model, args)
        print(f"AUC : {auc}, ACC : {acc}")

        return preds

    def test(self, args, model, test_data):
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)

        total_preds = []
        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args)

            preds = model(input)

            # predictions
            preds = preds[:,-1]

            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()
                
            total_preds.append(preds)

        total_preds = np.concatenate(total_preds)
            
        return total_preds

    def get_target(self, datas):
        targets = []
        for data in datas:
            targets.append(data[0][-1])

        return np.array(targets)


class Stacking:
    def __init__(self, trainer):
        self.trainer = trainer


    def get_train_oof(self, args, data, fold_n=5, stratify=True):

        oof = np.zeros(data.shape[0])

        fold_models = []

        if stratify:
            kfold = StratifiedKFold(n_splits=fold_n)
        else:
            kfold = KFold(n_splits=fold_n)

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        target = self.trainer.get_target(data)
        for i, (train_index, valid_index) in enumerate(kfold.split(data, target)):
            train_data, valid_data = data[train_index], data[valid_index]

            # 모델 생성 및 훈련
            print(f'Calculating train oof {i + 1}')
            trained_model = self.trainer.train(args, train_data, valid_data)

            # 모델 검증
            predict = self.trainer.evaluate(args, trained_model, valid_data)
            
            # fold별 oof 값 모으기
            oof[valid_index] = predict
            fold_models.append(trained_model)

        return oof, fold_models

    def get_test_avg(self, args, models, test_data):
        predicts = np.zeros(test_data.shape[0])

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        for i, model in enumerate(models):
            print(f'Calculating test avg {i + 1}')
            predict = self.trainer.test(args, model, test_data)
              
            # fold별 prediction 값 모으기
            predicts += predict

        # prediction들의 average 계산
        predict_avg = predicts / len(models)

        return predict_avg


    def train_oof_stacking(self, args_list, data, fold_n=5, stratify=True):
    
        S_train = None
        models_list = []
        for i, args in enumerate(args_list):
            print(f'training oof stacking model [ {i + 1} ]')
            train_oof, models = self.get_train_oof(args, data, fold_n=fold_n, stratify=stratify)
            train_oof = train_oof.reshape(-1, 1)

            # oof stack!
            if not isinstance(S_train, np.ndarray):
                S_train = train_oof
            else:
                S_train = np.concatenate([S_train, train_oof], axis=1)

            # store fold models
            models_list.append(models)

        return models_list, S_train

    def test_avg_stacking(self, args, models_list, test_data):
    
        S_test = None
        for i, models in enumerate(models_list):
            print(f'test average stacking model [ {i + 1} ]')
            test_avg = self.get_test_avg(args, models, test_data)
            test_avg = test_avg.reshape(-1, 1)

            # avg stack!
            if not isinstance(S_test, np.ndarray):
                S_test = test_avg
            else:
                S_test = np.concatenate([S_test, test_avg], axis=1)

        return S_test


    def train(self, meta_model, args_list, data):
        models_list, S_train = self.train_oof_stacking(args_list, data)
        target = self.trainer.get_target(data)
        meta_model.fit(S_train, target)
        
        return meta_model, models_list, S_train, target

    def test(self, meta_model, models_list, test_data):
        S_test = self.test_avg_stacking(args, models_list, test_data)
        predict = meta_model.predict(S_test)

        return predict, S_test
