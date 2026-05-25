# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

학생 구현 대상:
- ReLU.forward, ReLU.backward
- Softmax.forward, Softmax.backward
"""

import numpy as np


class ReLU:
    def __init__(self):
        self.mask = None
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    """

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        # 구현 완료: ReLU는 0 이하를 꺼야 하므로, 먼저 꺼질 위치를 표시한다.
        # 0 이하인 위치를 꺼둘 자리로 표시해 둔다.
        self.mask = x <= 0
        # 원본 x를 바로 바꾸지 않으려고 복사본을 만든다.
        out = x.copy()
        # mask가 True인 위치, 즉 0 이하였던 값은 ReLU 규칙대로 0으로 만든다.
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        # 구현 완료: 순전파 때 꺼진 위치는 역전파 때도 그대로 막는다.
        # forward 때 꺼졌던 위치는 backward에서도 gradient를 0으로 막는다.
        dout[self.mask] = 0
        # 남은 gradient를 앞 층으로 넘길 값으로 둔다.
        dx = dout
        return dx

class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다.
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """
        # 구현 완료: row별 max를 빼고 exp를 계산해서 큰 숫자로 터지는 상황을 줄인다.
        # 각 데이터 줄에서 최댓값을 빼서 exp 계산이 터지지 않게 만든다.
        shifted = x - np.max(x, axis=1, keepdims=True)
        # softmax 공식의 exp 부분을 계산한다.
        exp_x = np.exp(shifted)
        # exp 값을 각 줄의 exp 합으로 나눠서 확률처럼 만든다.
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return out

    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분한 gradient를 train()에서 직접 만들기 때문에
        여기서는 받은 gradient를 그대로 통과시킵니다.
        """
        # 구현 완료: train()에서 만든 gradient를 그대로 반환한다.
        # 이 과제에서는 softmax+loss gradient를 밖에서 만들기 때문에 그대로 넘긴다.
        return dout
