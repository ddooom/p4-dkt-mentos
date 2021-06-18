# HyperParameter 정리

## Loss

[참고 자료](https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)

이진 분류 문제로 풀어야 한다.

- 0 ~ 1 의 확률 값이 preds로 들어가고
- [0, 1]의 값이 target으로 들어간다.

### DiceBCELoss

### LabelSmoothingLoss

## Optimizer

### SGD

### adam

### adamW

### adamP

### madgrad

## Scheduler

### CyclicLR

### CosineAnnealingLR

### CosineAnnealingWarmRestarts

### CosineAnnealingWarmupRestarts

### StepLR

### get_linear_schedule_with_warmup

트랜스포머에서 새롭게 만든 WramUp 스케쥴러. 기존 WramUp과 달리 초깃값이 0이 아니고 살짝 높다.

## PopulationBasedTraining

### time_attr

시간 비교하는데 사용, meausre of progress

### metric

Stopping procedure will use this attrbute

### mode

Determines whether objective is minimizing or maximizing the metric attribute.

### perturbation_interval

Models will be considered for perturbation at this interval of `time_attr`. Note that perturbation incurs checkpoint overhead, so you shouldn't set this to be too frequent.

> 너무 작은 값을 사용하면 안 될 듯

### hyperparam_mutations

HyperParams to muatate.

`hyperparam_mutations` 또는 `custom_explore_fn`중 하나 이상을 지정해야 한다.

Tune will use the search space provided by `hyperparam_mutations` for the initial samples if the corresponding attributes are not present in `config`.

### quantile_fraction

0 ~ 0.5 값을 설정하라고 하는데... 정확히 뭐 하는 건지 모르겠다.

0일 경우는 exploitation이 전혀 수행되지 않는다고 한다.

> 아 이거 그거네, 하위 몇 %를 상위 몇 %로 바꾸는지

### resample_probability

hyperparam_mutions를 적용할 때 원래 분포에서 다시 샘플링할 확률. 다시 샘플링하지 않을 경우, 이 값은 **연속일 경우 1.2 또는 0.8의 인자에 의해 교란되거나 이산일 경우 인접 값으로 변경됩니다.**

> 원래 분포에서 다시 샘플링 해서 적용할 수 도 있구나...

### custom_explore_fn

이건 사용하지 말자

### require_attrs

Whether to require time_attr and metric to appear in result for every iteration. If True, error will be raised if these values are not present in trial result.

> True로 해야지 Stopper가 잘 동작할 것 같은데 ~ 

### synch

- False인 경우 PBT를 비동기로 수행
- True인 경우 PBT를 동기로 수행

Default: False

> 엥?? 비동기로 수행해도 되나.. 지금까지 비동기로 한 거 같은데

> 공유 데이터 저장소를 읽고 쓰는 기능만 추가함으로써 PBT를 수행할 수 있다. False로 하면 된다.

