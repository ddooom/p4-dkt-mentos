# DKT-TUTORIAL

DKT TASK를 유연하게 실험할 수 있는 Repository입니다.

## Install

```
pip install rich
pip install 'ray[tune]'
pip install easydict
```

## Usage

Jupyter Notebook 환경에서 자유롭게 실험하는 것을 권장합니다.

### Features

module path : `./fe/...`

```text
FEBase.save_feature_df()  # Cache에 Feature를 저장합니다.
FEBase.load_feature_df()  # 저장된 Feature를 불러옵니다.
FEBase.transform()        # Feature를 생성합니다.

SeqFeBase.transform()     # Sequence Feature를 생성합니다.
AggFeBase.transform()     # Aggregation Feature를 생성합니다.

FEPipeline.description()  # 생성되는 Feature들의 Description입니다.
FEPipeline.debug()        # Feature의 선행 순서를 파악하여 작동 유무를 설명합니다.
FEPipeline.transform()    # Feature들을 생성합니다. 
```

**사용 방법**

```python
from fe.feature import FEPipeline
from fe.agg import MakeCorrectCount, MakeCorrectPercent, MakeQuestionCount
from fe.seq import SplitAssessmentItemID, MakeFirstClass, MakeSecondClass, ConvertTime

fe_pipeline = FEPipeline(
    args, [
        SplitAssessmentItemID,
        ConvertTime,
        MakeFirstClass,
        MakeSecondClass,
        MakeCorrectCount,
        MakeQuestionCount,
        MakeCorrectPercent,
        MakeDifficultyByFirstClass
    ]
)
```

![image](https://user-images.githubusercontent.com/40788624/122508514-117e3b80-d03d-11eb-9fcb-aa017eebe40e.png)

[**Logging Examples**](./logging_examples/features.log)

### Preprocessing

module path : `./dkt_dataset.py`

```text
Preprocess.feature_engineering()  # Train, Test 데이터 셋의 Feature Engineering을 진행합니다.
Preprocess.split_data()           # Train 데이터셋에서 Validation 데이터셋을 만듭니다.
Preprocess.scaling()              # 각 Feature들을 Scaling합니다. (LabelEncoder, MinMaxScaler, StandardScaler)
Preprocess.data_augmentation()    # 학습 데이터를 GroupBy 메서드 및 테스트 데이터를 사용하여 증강합니다.
```
**사용 방법**

```python
columns = ["userID", "answerCode", "testPaper", "timeSec", "firstClass", 
            "secondClass", "correctPer", "firstClassDifficulty"]

pre_encoders = {
    "label": ["testPaper", "firstClass", "secondClass"],
    "min_max": ["correctPer"],
    "std": ["timeSec", "firstClassDifficulty"],
}

preprocess = NewSplitPreprocess(args, fe_pipeline, columns)
```

![image](https://user-images.githubusercontent.com/40788624/122508393-e693e780-d03c-11eb-85f5-eca5b12cdbf2.png)

[**Logging Examples**](./logging_examples/preprocess.log)

### Trainer

module path : `./trainer.py`

```text
Trainer._helper_init()    # 로깅 디렉토리 생성
Trainer._save_config()    # Config 저장
Trainer._get_model()      # 모델 생성 및 초기화
Trainer._collate_fn()
Trainer._get_loaders()

Trainer._save_model()     # 모델 저장
Trainer._load_model()     # 모델 로드
Trainer._get_metric()     # 성능 평가
Trainer._compute_loss()   # 로스 계산
Trainer._precess_batch()  # 배치 처리
Trainer._update_params()  # 모델 업데이트

Trainer._hyper()          # 하이퍼 파라미터 튜닝
Trainer._train()          # 학습
Trainer._validate()       # 검증
Trainer._inference()      # 추론

Trainer.hyper()           # 하이퍼 파라미터 튜닝
Trainer.run()             # 학습, 검증, 추론
Trainer.run_cv()          # Cross Validation
Trainer.debug()           # Debug
```

#### Custom Trainer

Method를 바꿈으로써 Custom Trainer를 작성 할 수 있습니다.

**Feature Trainer** : 어떤 Feature가 들어와도 유연하게 작동하는 Trainer 

```python
class FeatureTrainer(DKTTrainer):
    def _process_batch(self, batch):
        batch['mask'] = batch['mask'].type(torch.FloatTensor)
        batch["answerCode"] = batch["answerCode"].type(torch.FloatTensor)

        batch["interaction"] = batch["answerCode"] + 1
        batch["interaction"] = batch["interaction"].roll(shifts=1, dims=1)
        batch["mask"] = batch["mask"].roll(shifts=1, dims=1)
        batch["mask"][:, 0] = 0
        batch["interaction"] = (batch["interaction"] * batch["mask"]).to(torch.int64)
        
        
        for k in self.args.n_linears: # 수치형
            batch[k] = batch[k].type(torch.FloatTensor)
            
        for k, v in self.args.n_embeddings.items(): # 범주형
            batch[k] = batch[k].to(torch.int64)
            
        for k in batch.keys():
            batch[k] = batch[k].to(self.args.device)
        
        return batch
```

**Loss Shift Trainer** : Loss 전달 방식을 수정한 Trainer

```python
class Loss1Trainer(FeatureTrainer):  # FeatureTrainer를 상속
    def _collate_fn(self, batches):
        """ key값으로 batch 형성 """
        new_batches = {k: [] for k in batches[0].keys()}

        # batch의 값들을 각 column끼리 그룹화
        for k in batches[0].keys():
            for batch in batches:
                pre_padded = torch.zeros(self.args.max_seq_len)
                pre_padded[-len(batch[k]) :] = batch[k]
                new_batches[k].append(pre_padded)

        for k in batches[0].keys():
            new_batches[k] = torch.stack(new_batches[k])

        return new_batches


class Loss3ShiftTrainer(Loss1Trainer):
    def _compute_loss(self, preds, targets):
        loss = get_criterion(preds, targets)
        loss = loss[:, -3:]
        loss = torch.mean(loss)
        return loss


class Loss5ShiftTrainer(Loss1Trainer):
    def _compute_loss(self, preds, targets):
        loss = get_criterion(preds, targets)
        loss = loss[:, -5:]
        loss = torch.mean(loss)
        return loss
```


**LSTM Trainer** : 실험이 완료된 후, Module에 LSTM 기반으로 작성한 Trainer

- 여기서는 코드를 보고도 어떻게 동작하는 지 알 수 있게 최대한 가독성 있게 작성합니다.

```python
class LSTMTrainer(DKTTrainer):
    def _process_batch(self, batch):
        batch["mask"] = batch["mask"].type(torch.FloatTensor)
        batch["answerCode"] = batch["answerCode"].type(torch.FloatTensor)
        batch["correctPer"] = batch["correctPer"].type(torch.FloatTensor)
        batch["timeSec"] = batch["timeSec"].type(torch.FloatTensor)

        batch["interaction"] = batch["answerCode"] + 1
        batch["interaction"] = batch["interaction"].roll(shifts=1, dims=1)
        batch["mask"] = batch["mask"].roll(shifts=1, dims=1)
        batch["mask"][:, 0] = 0
        batch["interaction"] = (batch["interaction"] * batch["mask"]).to(torch.int64)

        batch["testPaper"] = batch["testPaper"].to(torch.int64)
        batch["firstClass"] = batch["firstClass"].to(torch.int64)
        batch["secondClass"] = batch["secondClass"].to(torch.int64)

        for k in batch.keys():
            batch[k] = batch[k].to(self.args.device)

        return batch
```

**사용 방법**

```python
trainer = Loss1Trainer(args, LSTM)   # LSTM Model with Loss1Trainer
auc, acc = trainer.run_cv(train_dataset, valid_dataset, test_dataset,
                         folds=5, seeds=[0, 1, 2, 3, 4])
clear_output()
print(f"auc: {auc}, acc: {acc}")
```

[**Logging Examples**](./logging_examples/LOG_[06.10_11:54])
