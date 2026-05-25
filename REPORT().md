# MNIST Handwritten Digit Recognition Report

## 0. 반·팀원

| 항목 | 내용 |
| --- |  |
| 반 | 303 |
| 팀명 | 최현진 |
| 팀원 | 정영훈 |
| 팀원 | 김다애 |
## 1. 실험 목적

본 과제의 목적은 PyTorch, TensorFlow 같은 딥러닝 프레임워크 없이 NumPy만 사용해 MNIST 손글씨 숫자 분류기를 구현하는 것이다.

구현 과정에서 단순히 정확도를 높이는 것보다, 다음 학습 흐름을 직접 구현하고 설명할 수 있는지를 중점으로 두었다.

```text
Forward -> Loss -> Backward -> Optimizer Update
```

최종 목표는 MNIST 테스트 정확도 95% 이상, 권장 97% 이상을 달성하는 것이다.

## 2. 모델 구조

주 제출 모델은 완전연결 다층 퍼셉트론(MLP) 구조를 사용하였다.

```text
Input: 784
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(10)
-> Softmax
```

| 구성 요소 | 내용 |
| --- | --- |
| 입력층 | 28x28 이미지를 펼친 784차원 벡터 |
| 은닉층 | 512, 256 |
| 출력층 | 10개 클래스 |
| 활성화 함수 | ReLU |
| 출력 함수 | Softmax |
| 정규화 | BatchNorm |
| 규제 | Dropout |
| 초기화 | 은닉층 He 초기화, 출력층 Xavier 계열 초기화 |

추가 실험에서는 같은 MLP 구조에 학습 중 1픽셀 랜덤 shift augmentation을 적용하였다.

## 3. 학습 설정

주 제출 모델 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| Optimizer | Adam |
| 초기 learning rate | 0.001 |
| learning rate schedule | epoch 16: 0.0005, epoch 22: 0.0002 |
| epochs | 25 |
| batch size | 128 |
| Dropout ratio | 0.1 |
| BatchNorm momentum | 0.9 |
| loss | Cross Entropy |

추가 실험의 최고 기록은 다음 조건에서 나왔다.

| 항목 | 값 |
| --- | --- |
| Augmentation | train batch마다 1픽셀 랜덤 shift |
| Dropout ratio | 0.1 |
| Weight decay | 5e-5 |
| best epoch | 19 |
| validation 기준 선택 | 사용 |

## 4. 실험 환경

| 항목 | 내용 |
| --- | --- |
| 실행 환경 | 로컬 Windows |
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| Matplotlib | 사용 |
| pytest | 사용 |
| GPU | MNIST 구현은 NumPy 기반이라 CUDA/GPU 사용 없음 |

대표 실행 시간:

| 실험 | 시간 |
| --- | --- |
| 기본 15 epoch 전체 학습 | 약 68.87초 |
| 25 epoch 병렬 sweep 중 최고 후보 | 약 543.84초 |
| augmentation 포함 추가 실험 최고 후보 | 약 2546.0초 |

augmentation 실험은 Python/NumPy에서 이미지 이동을 batch마다 처리했기 때문에 기본 학습보다 시간이 크게 증가하였다.

## 5. 결과

### 5.1 단위 테스트

전체 테스트를 실행하였다.

```bash
python -m pytest tests/ -v
```

결과:

```text
21 passed
```

### 5.2 주 제출 모델 결과

템플릿 `src/` 기준으로 재현 가능한 주 제출 모델 결과는 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 모델 | 784 -> 512 -> 256 -> 10 |
| BatchNorm | 사용 |
| Dropout | 0.1 |
| test accuracy | 98.73% |
| best epoch | 24 |
| final accuracy | 98.68% |
| final train loss | 0.001453 |
| 총 파라미터 수 | 537,354 |

Loss curve:

![MNIST base loss curve](report_assets/mnist_base_loss_curve.png)

Accuracy curve:

![MNIST base accuracy curve](report_assets/mnist_base_accuracy_curve.png)

### 5.3 추가 실험 결과

1픽셀 랜덤 shift augmentation을 적용한 실험에서는 더 높은 정확도를 얻었다.

| 항목 | 값 |
| --- | --- |
| 모델 | 784 -> 512 -> 256 -> 10 |
| BatchNorm | 사용 |
| Dropout | 0.1 |
| Augmentation | 1픽셀 랜덤 shift |
| Weight decay | 5e-5 |
| best validation accuracy | 99.00% |
| test accuracy at best validation | 99.08% |
| best epoch | 19 |
| 총 파라미터 수 | 537,354 |

Augmentation 실험 곡선:

![MNIST augmentation curve](report_assets/mnist_aug_loss_val_curve.png)

## 6. 회고

학습 손실은 epoch가 진행될수록 안정적으로 감소하였다. 기본 MLP 모델만으로도 권장 목표인 97%를 넘겨 98.73%를 달성하였다.

정확도 향상에 가장 효과적이었던 시도는 모델을 무작정 키우는 것이 아니라, Dropout 비율을 0.2에서 0.1로 낮추고 learning rate decay를 적용한 것이었다. 더 넓거나 깊은 MLP는 파라미터 수가 늘었지만 정확도 향상은 크지 않았다.

추가 실험에서 1픽셀 랜덤 shift augmentation을 적용하자 99.08%까지 정확도가 올라갔다. 이는 MNIST 숫자가 약간 이동해도 같은 숫자라는 성질을 모델이 더 잘 학습했기 때문으로 해석된다.

과대적합 관점에서는 test set을 반복해서 보는 방식이 바람직하지 않으므로, 추가 실험에서는 train 데이터 일부를 validation set으로 분리하고 validation accuracy가 가장 높은 시점의 모델을 test set에서 확인하였다.

이번 과제를 통해 Forward, Loss, Backward, Optimizer Update가 각각 따로 존재하는 함수가 아니라 하나의 학습 루프로 연결된다는 점을 확인하였다. 특히 BatchNorm과 Dropout은 train/test 모드에서 동작이 달라지므로, 평가 시 `train=False`로 전환하는 것이 중요했다.

향후 개선 방향으로는 CNN 구조를 직접 구현하는 방법이 있다. 참고 논문에서는 Convolution, Max Pooling, BatchNorm을 결합한 간단한 CNN으로 99.22%를 보고하였다. 다만 이번 과제의 기본 TODO 범위는 MLP 구현이므로, CNN은 추가 확장 과제로 보는 것이 적절하다.
