# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # 구현 완료: 정답 칸의 예측 확률만 뽑아 -log 벌점의 batch 평균을 계산한다.
    # y_pred의 줄 개수, 즉 이번에 한 번에 들어온 데이터 개수를 구한다.
    batch_size = y_pred.shape[0]
    # log(0)이 되지 않도록 예측 확률을 안전한 범위로 정리한다.
    probs = np.clip(y_pred, 1e-7, 1.0)
    # 각 데이터 줄에서 정답 label 칸의 확률만 뽑아 batch_size 길이로 다시 묶는다.
    correct_probs = probs[np.arange(batch_size), y_true]
    # 정답 확률에 -log를 씌운 벌점을 만들고, batch 평균 loss로 만든다.
    loss = -np.mean(np.log(correct_probs))
    return loss
