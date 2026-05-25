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
        # TODO: backward에서 사용할 입력 x를 저장하고 x @ W + b를 반환하세요.
        # x @ W + b -> @: 파이썬 행렬 곱셈(내적) 연산자 == np.dot(x, w) + b 
        # 입력 x 저장
        self.x = x

        # x @ W + b 반환 
        out = x @ self.W + self.b

        return out
        #raise NotImplementedError("Affine.forward를 구현하세요.")

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)

        Returns:
            dx: (batch_size, input_dim)

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
        """
        # TODO: self.dW, self.db, dx를 계산하세요.
        # 힌트: dW = x.T @ dout, db = batch 방향 합, dx = dout @ W.T 
        # 가중치 행렬(W)에 대한 최종 손실 함수(L)의 미분값(기울기)
        self.dW = self.x.T @ dout

        # 편향(b)에 대한 최종 손실 함수(L)의 미분값(기울기)
        self.db = np.sum(dout, axis=0)

        # 입력 데이터(x)에 대한 최종 손실 함수(L)의 미분값(기울기)
        dx = dout @ self.W.T

        return dx
    
        # raise NotImplementedError("Affine.backward를 구현하세요.")


class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """
    """
    데이터 분포 평균: 0
    데이터 분산: 1로 정규화
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터 -> 확대 담당, 1부터 시작
            beta: 정규화된 값에 더하는 shift 학습 파라미터 -> 이동 담당, 0부터 시작 
            momentum: running_mean/running_var 이동평균 비율 -> 추론 단계에서 사용할 이동 평균(running_mean), 이동 분산(running_var)을 계산할 때 과거의 값을 얼마나 반영할지 결정하는 비율 
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta) # 추론 때 사용할 feature별 평균의 이동평균 μ
        self.running_var = np.zeros_like(beta) # 추론 때 사용할 feature별 분산의 이동평균 σ
        self.eps = 1e-7 # epsilon 엡실론(작은 값) -> 0을 나누는 사태를 예방

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용
            
            배치 통계: 지금 현재 신경망을 통과하고 있는 해당 미니 배치 데이터들만을 계산하여 얻은 평균과 분산
            -> 학습 중 현재 미니 배치 데이터를 정규화하는 데 사용함(평균 0, 분산 1 만들기)

            running 통계: 학습 시작부터 지금까지 지나쳐간 모든 미니배치 통계치를 이동 평균 방식으로 누적해 놓은 전체 데이터의 추정치 
            -> running은 학습 중 데이터 정규화를 위해 쓰이지 않음! -> 값만 계속 갱신하고 추론 단계에서 사용함 
        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        # TODO: train=True에서는 batch mean/var로 정규화하고 running 통계를 갱신하세요.
        # TODO: train=False에서는 running_mean/running_var를 사용하세요.
        if train:
            # batch mean/var 정규화
            # 현재 배치에서 평균 구하기
            mean = np.mean(x, axis = 0)
            
            # 현재 배치에서 분산 구하기
            var = np.var(x, axis = 0)

            # 현재 배치 값에서 평균을 빼서 데이터를 0 중심으로 옮기기
            self.x_centered = x - mean

            # 표준편차로 나누기 위한 값 미리 계산
            # sqrt: square root, 제곱근
            # 분산 var의 제곱근 = 표준편차
            # 왜 1.0에 sqrt(var + self.eps) 나눔
            # x_norm = (x - mean) / np.sqrt(var + self.eps) -> 이렇게 한 줄로 끝낼 수도 있지만 backward에서 재사용하기 위해 분리 
            self.std_inv = 1.0 / np.sqrt(var + self.eps)

            # 정규화된 값: 평균 0, 분산 1
            # 이걸 왜 따로 안 쓰고 곱해줌?
            self.x_norm = self.x_centered * self.std_inv

            # 정규화한 값을 gamma로 scale(확장 = 곱), beta로 shift(이동 = 더하기)
            out = self.gamma * self.x_norm + self.beta

            # backward에서 다시 쓸 값 저장
            self.x = x
            self.mean = mean
            self.var = var
            self.batch_size = x.shape[0] # 현재 미니 배치 x에 들어있는 샘플 개수 

            # running 통계 갱신
            # new running_ = 
            # new running_ = old running_을 momentum % 유지 + 현재 running_을 1-momentum % 반영
            # v는 m과 다르게 방향이 아니라 크기 기록 -> 제곱으로 부호 날리고 크기를 확인  
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            return out
        
        # train == False
        # train 단계가 아닌 추론 단계일 때 -> 현재 배치 통계가 아닌 누적된 running 통계 사용
        x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # gamma로 scale(확장 = 곱), beta로 shift(이동 = 더하기)
        out = self.gamma * x_norm + self.beta
        
        return out 
        # raise NotImplementedError("BatchNorm.forward를 구현하세요.")

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        # TODO: self.dbeta, self.dgamma, dx를 계산하세요.
        # 힌트: 먼저 dbeta와 dgamma shape가 beta/gamma와 같은지 확인합니다.
        # beta는 shift 덧셈 노드 계산이라 역전파로 입력 값 dout을 그대로 보냄 
        # -> feature별 파라미터라 batch 방향으로 합쳐야 해서 np.sum 사용 
        self.dbeta = np.sum(dout, axis=0)

        # gamma는 scale 곱셈 노드 계산이라 상류의 값에 순전파 때의 입력 신호들을 서로 바꿔 곱해서 보냄
        # -> forward에서 self.x_norm랑 곱해졌음
        # -> feature별 파라미터라 batch 방향으로 합쳐야 해서 np.sum 사용 
        self.dgamma = np.sum(dout * self.x_norm, axis=0)

        # 정규화 과정 진행
        # -> x_norm = (x - mean) / np.sqrt(var + eps)를 거꾸로 진행
        N = self.batch_size

        # x_norm에 대한 기울기를 구함
        # forward에서 self.gamma랑 곱해졌음
        #
        dx_norm = dout * self.gamma

        # x_norm = x_centered * std_inv 에서 std_inv가 받은 기울기
        dstd_inv = np.sum(dx_norm * self.x_centered, axis=0)

        # x_norm = x_centered * std_inv 에서 x_centered로 바로 넘어가는 기울기
        dx_centered1 = dx_norm * self.std_inv

        # std_inv = 1 / sqrt(var + eps) 에서 var가 받은 기울기
        # 왜 -0.5 곱하고 왜 -1.5제곱하는지 
        # std_inv = 1 / sqrt(var + eps)
        # -> std_inv = (var + eps)^(-1/2)
        # -> var로 미분 = -1/2 * (var + eps)^(-3/2) => -0.5, -1.5제곱이 나옴 
        dvar = dstd_inv * (-0.5) * (self.var + self.eps) ** (-1.5)

        # var = np.mean((x - mean)^2) 에서 x_centered로 추가로 넘어가는 기울기
        # var = mean(x_centered ** 2)를 거꾸로 미분
        dx_centered2 = (2.0 / N) * self.x_centered * dvar

        # x_centered로 들어온 두 갈래의 기울기를 합침
        dx_centered = dx_centered1 + dx_centered2

        # x_centered = x - mean 에서 mean가 받은 기울기.
        dmean = -np.sum(dx_centered, axis=0)

        # mean = 모든 mean(x) 이므로, mu의 기울기를 batch의 각 x에 고르게 나눠줌
        dx_mean = dmean / N

        # 최종적으로 이전 층으로 넘길 입력 x의 기울기.
        dx = dx_centered + dx_mean

        return dx
        # raise NotImplementedError("BatchNorm.backward를 구현하세요.")


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
        # TODO: train=True에서는 mask를 만들고 x에 곱하세요.
        # TODO: train=False에서는 x * (1 - drop_ratio)를 반환하세요.
        raise NotImplementedError("Dropout.forward를 구현하세요.")

    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        # TODO: forward에서 만든 mask를 dout에 곱하세요.
        raise NotImplementedError("Dropout.backward를 구현하세요.")
