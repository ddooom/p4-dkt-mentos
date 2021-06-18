# LGBM with Row data
### Input arguments
- `y_id` : y를 결정할 때 userID 별로 y를 결정할 것인지, Cycle 별로 y를 결정할 것인지 선택
    (cycle : 한 사용자가 푼 하나의 test, 한 사용자가 같은 test를 두 번 풀어도 두 test는 다른 cycle)
- `y_method` : y를 y_id의 마지막 행 answerCode로 할 것인지, 다음 행의 answerCode로 할 것인지 선택 
- `n_folds` : k-fold를 할 때, fold의 수

### Feature Engineering
- 'Timestamp' : 기존 Timestamp를 초 단위로 변경
- 'time_diff' : cycle 단위로 문제 푼 시간
- 'UserCumtestnum' : userID 별 푼 문제 수
- 'UserTagCumtestnum' : userID, KnowledgeTag 별 누적 푼 문제 수
- 'UserTestCumtestnum' : userID, testId 별 누적 푼 문제 수
- 'TestSize' : testId 별 문제 수
- 'Retest' : userID 별로 testId를 푼 횟수 (한 user가 특정 testId를 처음 풀면 0, 두 번째 풀면 1)
- 'UserCycleCumtestnum' : cycle 별 누적 푼 문제 수
- 'testNumber' : test 별 문제 번호 (assessmentItemID의 뒤 3글자)
- 'UserCumcorrectnum' : userID 별 누적 맞춘 문제 수
- 'UserCumcorrectper' : userID 별 누적 정답률
- 'UserTagCumcorrectnum' : userID, KnowledgeTag 별 누적 맞춘 문제 수
- 'UserTagCumcorrectper' : userID, KnowledgeTag 별 누적 정답률
- 'UserTestCumcorrectnum' : userID, testId 별 누적 맞춘 문제 수
- 'UserTestCumcorrectper' : userID, testId 별 누적 정답률
- 'UserCycleCumcorrectnum' : cycle 별 누적 맞춘 문제 수
- 'UserCycleCumcorrectper' : cycle 별 누적 정답률
- 'quesCnt' : user 별 푼 문제 수
- 'correctCnt' : user 별 맞춘 문제 수
- 'correctPer' : user 별 정답률
- 'top10CorrectPer' : user 별 마지막 10 문제 정답률
- 'top30CorrectPer' : user 별 마지막 30 문제 정답률
- 'top50CorrectPer' : user 별 마지막 50 문제 정답률
- 'top100CorrectPer' : user 별 마지막 100 문제 정답률
- 'TagCorrectPer' : KnowledgeTag 별 정답률
- 'TagCorrectSum' : KnowledgeTag 별 맞춘 문제 수
- 'UserTagCorrectPer' : userID, KnowledgeTag 별 정답률
- 'UserTagCorrectSum' : userID, KnowledgeTag 별 맞춘 문제 수
- 'TestCorrectPer' : testId 별 정답률
- 'TestCorrectSum' : testId 별 맞춘 문제 수
- 'UserTestCorrectPer' : userID, testId 별 정답률
- 'UserTestCorrectSum' : userID, testId 별 맞춘 문제 수
- 'AssessmentCorrectPer' : AssessmentItemID 별 정답률
- 'AssessmentCorrectSum' : AssessmentItemID 별 맞춘 문제 수

(FE 과정 중 건모님의 FE class를 사용하였습니다. gdevelop branch 참고)

### Cross Validation
- y를 기준으로 stratified K-fold 

### Training and Inference
- **Training** : Input Variables를 모두 입력하고 feature를 추가한 뒤, Training 셀 실행
- **Inference** : Training 한 뒤 Inference 셀 실행
     결과로 output 디렉토리에 `lgbm_{y_id}_{y_method}_{n_folds}.csv` 파일 생성 
- **Training using Pycaret** : feature를 추가한 뒤, Training using Pycaret 셀 실행
    inference 방법은 위와 동일

### LB score
- library : **lightgbm**, y_id : **cycle**, y_method : **next**, folds : **5** → **LB acc : 0.7043, LB auc : 0.7685**
- library : **lightgbm**, y_id : **cycle**, y_method : **last**, folds : **5** → **LB acc : 0.7097, LB auc : 0.7842**
- library : **pycaret**, y_id : **cycle**, y_method : **last**, folds : **5** → **LB acc : 0.7204, LB auc : 0.7893**