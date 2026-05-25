# -*- coding: utf-8 -*-
"""
신경망 layer 모음.

학생 구현 대상:
- Affine.forward, Affine.backward
- BatchNorm.forward, BatchNorm.backward
- Dropout.forward, Dropout.backward
"""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = xW + b 입니다.
    MNIST에서는 784개 픽셀 입력을 은닉층/출력층 차원으로 선형 변환하는 역할을 합니다.
    """

    def __init__(self, W, b):
        """가중치 W와 편향 b를 외부 params dict와 같은 배열 객체로 공유합니다."""
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        # 구현 완료: backward에서 다시 쓸 입력 x를 저장하고 x @ W + b를 반환한다.
        # backward에서 dW를 구할 때 다시 써야 하므로 입력 x를 저장한다.
        self.x = x
        # Affine 층의 순전파 식: 입력에 W를 곱하고 b를 더한다.
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
        """
        # 구현 완료: 이전 층으로 넘길 dx와, W/b를 수정할 dW/db를 계산한다.
        # 공식: dW = x.T @ dout, db = batch 방향 합, dx = dout @ W.T
        # 뒤에서 온 기울기(dout)를 W의 반대 방향(W.T)으로 보내, 이전 입력 x로 넘길 기울기 dx를 만든다.
        dx = np.dot(dout, self.W.T)
        # W를 얼마나 수정해야 하는지 나타내는 기울기다. 결과 shape가 W와 같도록 x.T @ dout 순서로 곱한다.
        self.dW = np.dot(self.x.T, dout)
        # b는 batch의 모든 데이터에 더해졌으므로, 각 데이터에서 온 기울기를 세로로 합친다.
        self.db = np.sum(dout, axis=0)
        return dx
class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var 이동평균 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용

        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        # 구현 완료: 학습 때는 현재 batch 통계, 추론 때는 누적 running 통계를 사용한다.
        if train:
            # 현재 배치에서 feature별 평균을 구한다.
            mu = np.mean(x, axis=0)

            # 현재 배치에서 feature별 분산을 구한다.
            var = np.var(x, axis=0)

            # 평균을 빼서 데이터를 0 중심으로 옮긴다.
            self.x_centered = x - mu

            # 표준편차로 나누기 위한 값을 미리 계산한다.
            self.std_inv = 1.0 / np.sqrt(var + self.eps)

            # 정규화된 값: 평균 0, 분산 1에 가까운 값.
            self.x_norm = self.x_centered * self.std_inv

            # 정규화한 값을 gamma로 scale하고 beta로 shift한다.
            out = self.gamma * self.x_norm + self.beta

            # backward에서 다시 쓸 값들을 저장한다.
            self.x = x
            self.mu = mu
            self.var = var
            self.batch_size = x.shape[0]

            # 추론 모드에서 쓸 running 평균/분산을 갱신한다.
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            return out

        # train=False일 때는 현재 배치 통계가 아니라 누적된 running 통계를 사용한다.
        x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        # 구현 완료: beta/gamma를 수정할 dbeta/dgamma와 이전 층으로 넘길 dx를 계산한다.

        # beta는 forward에서 그냥 더해졌으므로, 넘어온 기울기를 feature별로 합친다.
        self.dbeta = np.sum(dout, axis=0)

        # gamma는 forward에서 x_norm에 곱해졌으므로, dout * x_norm을 feature별로 합친다.
        self.dgamma = np.sum(dout * self.x_norm, axis=0)

        # 여기부터는 정규화 과정 x_norm = (x - mean) / sqrt(var + eps)를 거꾸로 풀어간다.
        N = self.batch_size

        # out = gamma * x_norm + beta 에서 x_norm 쪽으로 넘어가는 기울기.
        dx_norm = dout * self.gamma

        # x_norm = x_centered * std_inv 에서 std_inv가 받은 기울기.
        dstd_inv = np.sum(dx_norm * self.x_centered, axis=0)

        # x_norm = x_centered * std_inv 에서 x_centered로 바로 넘어가는 기울기.
        dx_centered1 = dx_norm * self.std_inv

        # std_inv = 1 / sqrt(var + eps) 에서 var가 받은 기울기.
        dvar = dstd_inv * (-0.5) * (self.var + self.eps) ** (-1.5)

        # var = mean((x - mu)^2) 에서 x_centered로 추가로 넘어가는 기울기.
        dx_centered2 = (2.0 / N) * self.x_centered * dvar

        # x_centered로 들어온 두 갈래의 기울기를 합친다.
        dx_centered = dx_centered1 + dx_centered2

        # x_centered = x - mu 에서 mu가 받은 기울기.
        dmu = -np.sum(dx_centered, axis=0)

        # mu = mean(x) 이므로, mu의 기울기를 batch의 각 x에 고르게 나눠준다.
        dx_mu = dmu / N

        # 최종적으로 이전 층으로 넘길 입력 x의 기울기.
        dx = dx_centered + dx_mu

        return dx

class Dropout:
    """
    Dropout.

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식을 사용합니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 뉴런 비율."""
        self.drop_ratio = drop_ratio

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 무작위 mask 적용, False면 평균적인 출력 크기로 scale
        """
        # 구현 완료: 학습 때는 랜덤 mask로 일부 값을 끄고, 추론 때는 평균 비율만큼 줄인다.
        if train:
            # x와 같은 모양의 랜덤 표를 만들고, drop_ratio를 넘은 위치만 살린다.
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            # True인 위치는 원래 값을 유지하고, False인 위치는 0으로 꺼진다.
            return x*self.mask
        # 테스트 때는 랜덤으로 끄지 않고, 학습 때 살아남는 평균 비율만큼 줄인다.
        return x * (1 - self.drop_ratio)



    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        # 구현 완료: forward 때 만든 mask를 dout에 곱해 꺼진 위치의 gradient를 막는다.
        # forward 때 꺼진 위치는 backward에서도 gradient가 흐르지 않게 같은 mask를 곱한다.
        return dout * self.mask
