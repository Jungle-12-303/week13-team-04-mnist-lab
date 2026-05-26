# -*- coding: utf-8 -*-
"""파라미터 업데이트 규칙을 모아 둔 optimizer 모듈."""

import numpy as np


class SGD:
    """
    확률적 경사하강법(SGD).

    가장 단순한 optimizer로, 각 파라미터를 gradient 반대 방향으로 lr만큼 이동합니다.
    """

    def __init__(self, lr=0.01):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률."""
        self.lr = lr

    def update(self, params, grads):
        """params dict의 모든 파라미터를 제자리(in-place)에서 갱신합니다."""
        # TODO: params[key]를 gradient 반대 방향으로 업데이트하세요.
        for param in params:
            params[param] -= self.lr * grads[param]
        # raise NotImplementedError("SGD.update를 구현하세요.")


class Adam:
    """
    Adam Optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터별 학습률을 조절합니다.
    MNIST 과제에서는 SGD보다 빠르게 손실이 내려가는지 비교해 볼 수 있습니다.
    """

    def __init__(self, lr=0.001):
        """Args: lr: Adam 업데이트의 기본 학습률."""
        self.lr = lr
        self.m, self.v = {}, {}
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        # TODO: m, v 이동평균과 bias correction을 사용해 params를 업데이트하세요.
        
        if self.t == 0:
            for param in params:
                self.m[param] = np.zeros_like(params[param])
                self.v[param] = np.zeros_like(params[param])
                M_hat = self.m[param]
                V_hat = self.v[param]
        self.t += 1
        for param in params:
            self.m[param] = self.beta1 * self.m[param] + (1-self.beta1) * grads[param]
            self.v[param] = self.beta2 * self.v[param] + (1-self.beta2) * grads[param] * grads[param]
            M_hat = self.m[param] / (1 - self.beta1 ** self.t)
            V_hat = self.v[param] / (1 - self.beta2 ** self.t)
            params[param] = params[param] - self.lr * M_hat / np.sqrt(V_hat + 1e-8)

        # raise NotImplementedError("Adam.update를 구현하세요.")
