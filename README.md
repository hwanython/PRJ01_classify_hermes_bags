## Kankas AI Hermes Bag Classification
- 본 코드의 목적은 다음과 같습니다.
  - 에르메스 가방의 6개 종류에 대한 분류 문제를 해결
    
## Data Classes
- hermes_kelly: 0 
- hermes_birkin: 1
- hermes_lindy: 2
- hermes_gardenparty: 3
- hermes_constance: 4
- hermes_picotin: 5


## Codebase
- 각 파일은 directory명에 맞도록 배치되어 있기 때문에 구조를 쉽게 이해할 수 있을 것입니다.
- 대략적인 codebase 구조는 아래와 같습니다.  

```
[v0.5.0]
  │  Readme.md  
  │  requirements.txt  
  ├─ configs		: 학습 관련된 구성 요소 e.g. 하이퍼파라미터, 데이터 경로, 클래스
  ├─ datasets		: dataset과 관련된 code들, e.g., dataloader 정의
  ├─ experiements	: 학습 모델(.pth)이 저장된 폴더
  ├─ libs				: network와 관련된 code들 network building
  ├─ tools				: Training 및 Test loop가 정의된 코드
  └─ utils				: 그 외 유틸들 e.g.,
```

## 사용법
### Train, Validation, Test 데이터 셋 분류
- 학습에 적용하기 위한 데이터 셋(Train/Validation)과 성능 평가를 위환 Test 데이터셋의 분류
- "filepath", "labels"을 index로 하는 csv 파일로 저장하여 분류
- Train:Validation:Test = 8:1:1 비율 선정
- 다음과 같은 커맨드로 수행됩니다.
```
    python utils/get_data_table.py --root '../datasets/raws'  --save_dir '../datasets/csv'
```
- Train, Validation, Test 데이터 셋 분류 완료된 데이터는 다음과 같은 형식을 가집니다.
```
[train.csv]
 - filepath : 저장된 이미지 파일의 경로
 - labels : 해당 이미지의 라벨
```

### 학습 방법
- 학습은 다음과 같은 커맨드로 수행됩니다.
```
    python tools/train.py
```

### 테스트 방법
- 학습은 다음과 같은 커맨드로 수행됩니다.
```
    python tools/test.py
```


### Remove small particle
- 작성 예정



