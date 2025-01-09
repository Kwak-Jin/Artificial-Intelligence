# Machine Learning
Some examples of machine learning techniques are posted in [Industrial AI and Automation repository](https://github.com/Kwak-Jin/IAIA/tree/master/Tutorial/TU_MachineLearning)  
Why Machine Learning?
- Dataset이 부족함/ 카메라의 경우 3차원 rgb 및 픽셀수로 인해서 딥러닝을 잘 활용할 수 있지만, 1차원 시계열 데이터는 딥러닝이 부적합할 수 있음
- 비용적인 측면
- 속도(실시간 성을 보장해야하거나, Training 시킬 시간 등)
    
### 어느 분야에서 Machine Learning이 쓰이는지
산업에서 Machine Learning은 자동화/분류/고장예지 및 건전성 관리 등 아주 광범위하게 쓰인다.     
해당 문서에서는 산업 기반/ 기계적인 역학 기반/ 센서 신호 처리 기반에 대해서 주로 다룰 것이다.     
1. 배터리 수명 예측
2. 코일
3. 기어-베어링 시스템
4. PCB 기판
5. LED 등
    
**Terminology** 
- Fault: Abnormal state of system
  - Mechanical: 변형, 피로 등
  - Electrical: 쇼트 등
  - 대부분(약 50%)의 산업 장비의 fault 및 failure는 회전기계/베어링에서 발생함
- Failure: Event from fault/ permanent interruption of system under standard operating condition
- Malfunction: temporary interrupt of system function
- Reliability: Ability to perform required functions for a certain period of time
  - MTTF: mean time to failure
  - lambda: rate of failure per time unit, MTTF 역수
- Safety: Ability not causing danger(Fault, failure, malfunctions)
- Availability: Probability the system will operate satisfactorily at any period of time

베어링의 고장 원인:
1. Flaking
2. Crack
3. Brinelling
4. Fretting
## Theory
### Preprocessing/데이터 전처리
데이터를 가공하기 쉽게 처리하기 위해서 전처리는 필수적이다. 산업의 경우 센서 등의 발달로 데이터를 뽑을 수 있으나, 노이즈, 이상치, 결측치 등 많은 에러사항이 있을 수 있다.
Feature extraction의 원활한 진행을 위해 미리 전처리 과정을 거친다. [참고](https://chaheekwon.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC%EC%9D%98-%EA%B0%9C%EB%85%90%EA%B3%BC-%EC%A4%91%EC%9A%94%EC%84%B1)
- Noise reduction
- normalize
- outlier
- missing value
- offset removal

### Feature Extraction
- Condition indicator
- time domain
- frequency domain
- time-frequency domain(e.g. STFT)

#### Time domain feature extraction

#### Frequency domain feature extraction

### Feature Selection/ Feature reduction
#### Principal component analysis (PCA)

#### Other dimension reduction technique
##### Locally Linear Embedding(LLE)
##### Multidimensional scaling(MDS)
##### Linear discriminant analysis(LDA)

### Machine Learning models
#### Support Vector Machine (SVM)
#### Decision Trees
#### Ensemble method and Random forest

### Fault Diagnosis

### Fault Prediction

