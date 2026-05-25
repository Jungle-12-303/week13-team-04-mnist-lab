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
        """
        params: 신경망의 매개변수를 보관하는 딕셔너리 변수(인스턴스 변수)
        params['W']: 가중치
        pramas['b']: 편향

        grads: 기울기를 보관하는 딕셔너리 변수(numerical_gradient()의 반환값)
        grads['W']: 가중치의 기울기
        grads['b']: 편향의 기울기    
        """
        # TODO: params[key]를 gradient 반대 방향으로 업데이트하세요.
        for key in params.keys():
            params[key] -= self.lr * grads[key]

        # raise NotImplementedError("SGD.update를 구현하세요.")


class Adam:
    """
    Adam Optimizer.

    gradient의 이동평균(m)과 제곱 이동평균(v)을 함께 사용해 파라미터별 학습률을 조절합니다.
    MNIST 과제에서는 SGD보다 빠르게 손실이 내려가는지 비교해 볼 수 있습니다.
    """
    """
    이동평균 m == momentum의 속도 v
    제곱 이동평균 v == adam grad의 기울기 제곱의 누적합 h

    일차 모멘텀용 계수 beta 1 = 0.9
    이차 모멘텀용 계수 beta 2 0.999
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """Args: lr: Adam 업데이트의 기본 학습률."""
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # 빈 중괄호랑 None의 차이
        # {}: 빈 딕셔너리 객체 -> 나중에 키를 추가할 수 있음 
        # None: 아무 값도 없음 -> 파라미터 이름별로 값을 저장해야 해서(딕셔너리 형태로 사용해야 해서) {}로 두는 게 좋음 
        self.m, self.v = {}, {}
        
        # None은 여러 객체에 한 번에 넣을 수가 없음
        # self.m, self.v = None, None 아니면 한 줄에 하나씩 넣어줘야 함 
        # self.m, self.v = None ->  TypeError: cannot unpack non-iterable NoneType object

        # 업데이트 횟수 기록 
        self.t = 0
        

    def update(self, params, grads):
        """Adam 공식에 따라 params dict의 모든 파라미터를 갱신합니다."""
        # TODO: m, v 이동평균과 bias correction을 사용해 params를 업데이트하세요.

        for key in params.keys():
            # np.zeros_like(params[key]): 기준 배열(params[key])과 같은 shape의 0으로 채워진 배열을 만듦 
            # Adam은 계속 호출되면서 이동평균을 누적해야 함 -> 항상 빈 배열로 만들면 안 됨
            # -> self.m에 없던 key가 새로 들어온 경우에만 빈 배열 만들어줌
            # self.m에서만 검사하는 이유!: m과 v를 항상 한 세트로 같이 만듦
            # -> 즉, m에 없으면 v에도 없는 키이기 때문에 하나만 검사해도 괜찮음 
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        # Adam 업데이트 실행 때마다 횟수 증가  
        self.t += 1

        for key in params.keys():
            # new m = old m을 beta1 % 유지 + 현재 grad를 1-beta1 % 반영
            # 1 - beat 하는 이유: 전체 비율을 1로 맞추기 위함
            # -> beta + grad 비율 => 1이 되도록!
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # new v = old v을 beta2 % 유지 + 현재 grad 제곱을 1-beta2 % 반영
            # v는 m과 다르게 방향이 아니라 크기 기록 -> 제곱으로 부호 날리고 크기를 확인  
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 처음 m, v가 0에서 시작하기 때문에 초반엔 m, v가 실제 평균보다 작게 나옴
            # -> 1 - beta ** t로 나눠서 보정함
            # beta를 t제곱 후 1에서 뺌-> 처음 t가 1일 때, 첫 m이 대략 0.1 * grad로 작게 만들어지면 다시 0.1로 나눠서 원래 grad 크기에 가깝게 보정
            m_correction = self.m[key] / (1 - (self.beta1 ** self.t))
            v_correction = self.v[key] / (1 - (self.beta2 ** self.t))
            
            # 보정된 방향 m을 크기 v로 나눠 파라미터별 이동량 조절 
            # v_correction에 1e-7 더해서 v가 0일 때 0으로 나누는 상황을 막음
            params[key] -= self.lr * m_correction / (np.sqrt(v_correction) + 1e-7)

        # raise NotImplementedError("Adam.update를 구현하세요.")
