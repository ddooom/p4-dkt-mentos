# LGBM 
### Input Variables
- id_column : aggregation 데이터 셋을 구축할 때, 기준이 되는 column이다. ['userID', 'userID_testIdHead', 'userId_YMD_testId'] 중 하나 선택 가능하며 userID_testIdHead는 유저 별 대분류를, userId_YMD_testId는 유저 별 cycle을 나타낸다.
- n_folds : k-fold를 할 때, fold의 수

### Feature Engineering
- ID : aggregation 데이터 셋의 기준
- y : 마지막 문제의 정답 or 오답
- QuesCnt : 문항 수
- CorrectCnt : 맞은 문항 수
- CorrectPer : 정답률
- CycleCnt : Cycle 수
- QuesPerCycle : Cycle 당 문항 수
- Max-MinTimestampDay : timestamp 끝과 처음의 차이 (일 단위)
- MeanCycleIntervalDay : Cycle 간 간격의 평균 (일 단위)
- TopNCorrectPer : 최근 N 개의 문항의 정답률
- LastCycleCorrectPer : 마지막 cycle의 정답률
- MeanTimeDiffinCycle : Cycle 내부에서 문항 간 timestamp의 평균 → 구해진 평균들의 cycle 간 평균 (한 userID에 여러 cycle이 존재하기 때문)

### Cross Validation
- y를 기준으로 stratified K-fold 

### Feature Selection
- RFE

### Training and Inference
- Input Variables를 모두 입력한 뒤, 모든 셀 실행
- inference 결과로 output 디렉토리에 lgbm_{id_column}_k{n_folds}.csv 파일 생성 

### LB score
- userID 기준 → LB ACC : 0.6801, LB AUC : 0.7269 
- 대분류 기준 → LB ACC : 0.6801, LB AUC : 0.7290
- Cycle 기준 → LB ACC : 0.6855, LB AUC : 0.7313 