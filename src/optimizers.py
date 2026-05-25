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
        # 구현 완료: params의 각 값을 gradient 반대 방향으로 한 칸 움직인다.
        # W1, b1 같은 파라미터 이름을 하나씩 꺼낸다.
        for key in params.keys():
            # 파라미터를 gradient 반대 방향으로 learning rate만큼 움직인다.
            params[key] -= self.lr * grads[key]


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

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        # 구현 완료: m, v 이동평균과 초반 보정을 써서 params를 업데이트한다.
        # m은 방향 기록, v는 크기 기록을 얼마나 천천히 따라갈지 정하는 비율이다.
        beta1, beta2 = 0.9, 0.999
        # 0으로 나누는 상황을 막기 위한 아주 작은 값이다.
        eps = 1e-8
        # Adam update를 몇 번째 하는지 세는 카운터다.
        self.t += 1
    
        for key in params.keys():
            if key not in self.m:
                # 이 파라미터용 m, v 기억 공간이 없으면 같은 모양의 0 배열로 처음 배정한다.
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # 이전 방향 기록에 현재 gradient를 조금 섞어서 새 방향 기록을 만든다.
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            # 이전 크기 기록에 현재 gradient 제곱을 조금 섞어서 새 크기 기록을 만든다.
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)

            # 0에서 시작해서 초반에 작게 나온 m을 step 수에 맞춰 키워 보정한다.
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            # 0에서 시작해서 초반에 작게 나온 v도 step 수에 맞춰 키워 보정한다.
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            # 방향 기록을 크기 기록으로 조절한 최종 이동량만큼 파라미터를 업데이트한다.
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
