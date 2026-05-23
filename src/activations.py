# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

학생 구현 대상:
- ReLU.forward, ReLU.backward
- Softmax.forward, Softmax.backward
"""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.

        # 넘파이 배열에 부등호 연산을 수행하면 배열의 원소 각각에 부등호 연산을 수행한 bool 배열 생성
        # 조건을 만족하면 True로 설정 -> x가 0보다 작거나 같은 원소의 위치를 True로 설정 
        self.mask = (x <= 0)
        # print(self.mask)

        # .copy(): 원본 배열의 데이터와 메모리 공간을 완전히 새롭게 복사하는 깊은 복사를 수행하는 메서드         
        # 입력 데이터 x를 그대로 복사해서 out에 저장 
        out = x.copy()

        # 넘파이 배열 self.mask에 True로 저장된 위치의 원소만 골라 그 값을 일괄적으로 0으로 덮어씌움 
        # 음수 값은 0으로 막는 역할 
        out[self.mask] = 0
        # print(self.mask)

        return out 
        # raise NotImplementedError("ReLU.forward를 구현하세요.")

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient -> 기울기 (역전파를 통해 구한 미분 값들의 묶음)

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        # TODO: forward에서 저장한 self.mask를 이용해 gradient가 흐를 위치만 남기세요.
        dout[self.mask] = 0

        dx = dout

        return dx
        #raise NotImplementedError("ReLU.backward를 구현하세요.")


class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다. -> logit: 신경망에서 소프트맥스 함수에 들어가기 직전의 정규화되지 않은 출력값 (= score)
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.

    Affine 계층의 연산을 막 마친 직후의 수치 -> 로짓이 소프트맥스 계층을 거쳐 총합 1이 되는 확률 값으로 변환 
    """

    def __init__(self):
        # softmax의 출력
        self.y = 0

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """ 
        # TODO: 수치 안정성을 위해 row별 max를 뺀 뒤 softmax 확률을 계산하세요.
        # 힌트: np.max(..., axis=1, keepdims=True), np.exp, np.sum을 사용합니다.
        # print("원본 Numpy 배열")
        # print(x)
        
        row_max = np.max(x, axis=1, keepdims=True)
        # print("row 별 max")
        # print(row_max)

        # print("x - max 값")
        out = x - row_max
        # print(out)

        exp_x = np.exp(out)
        # print("x - max 값으로 exp 연산")
        # print(exp_x)

        # 오버플로우 대책 
        sum_exp_x = np.sum(exp_x, axis=1)
        # print("exp 연산 값 총합")
        # print(sum_exp_x)

        # 확률 구할 때 사용할 요소가 속해있는 행렬의 행 구하기
        # reshape(데이터 개수, 1): 데이터 개수 만큼의 행과 1개의 열로 이뤄진 2차원 배열 형태로 브로드캐스팅 
        # sum_exp_x = sum_exp_x.reshape(sum_exp_x.size, 1)
        # 차원 자리에 -1을 지정하면 원래 배열의 원소 개수에 맞게 넘파이가 자동으로 크기를 계산해 맞춰줌 
        sum_exp_x = sum_exp_x.reshape(-1, 1)

        # 확률, 각 요소의 값을 자신이 속한 행의 총합으로 나누기
        self.y = exp_x / sum_exp_x
        # print("확률")
        # print(self.y)

        return self.y
        # raise NotImplementedError("Softmax.forward를 구현하세요.")

    def backward(self, dout):
        """
        Softmax와 Cross Entropy(교차 엔트로피)를 함께 미분한 gradient를 train()에서 직접 만들기 때문에
        여기서는 받은 gradient를 그대로 통과시킵니다.
        """
        # TODO: train()에서 만든 gradient를 그대로 반환하세요.

        return dout
    
        # raise NotImplementedError("Softmax.backward를 구현하세요.")
