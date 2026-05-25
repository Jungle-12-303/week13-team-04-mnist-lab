# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.

        layer_sizes = [784, 512, 256, 10]

        # self.params에는 optimizer가 고칠 실제 배열들을 이름표와 함께 넣는다.
        self.params = {}

        # layer_sizes의 앞뒤 숫자를 보고 W/b 모양을 자동으로 만든다.
        for i in range(1, len(layer_sizes)):
            fan_in = layer_sizes[i - 1]
            fan_out = layer_sizes[i]

            # He 초기화: ReLU를 쓰는 은닉층에서 값이 너무 커지거나 작아지는 것을 줄인다.
            scale = np.sqrt(2.0 / fan_in)

            # W_i는 이전 층 fan_in개 입력을 다음 층 fan_out개 출력으로 바꾸는 행렬이다.
            self.params[f"W{i}"] = scale * np.random.randn(fan_in, fan_out)

            # b_i는 각 출력 칸에 더해지는 편향이라 fan_out 길이만큼 만든다.
            self.params[f"b{i}"] = np.zeros(fan_out)

            # 마지막 출력층에는 BatchNorm을 붙이지 않고, 은닉층에만 gamma/beta를 둔다.
            if i < len(layer_sizes) - 1 and use_batchnorm:
                # gamma는 정규화된 값을 다시 키우거나 줄이는 값이다.
                self.params[f"gamma{i}"] = np.ones(fan_out)

                # beta는 정규화된 값을 다시 옮기는 값이다.
                self.params[f"beta{i}"] = np.zeros(fan_out)

        # forward 순서를 보장하려고 일반 dict가 아니라 OrderedDict에 layer를 줄 세운다.
        self.layers = OrderedDict()

        # 방금 만든 params 배열들을 실제 layer 객체에 연결한다.
        for i in range(1, len(layer_sizes)):
            # 각 구간은 먼저 Affine으로 선형 변환한다.
            self.layers[f"Affine{i}"] = Affine(self.params[f"W{i}"], self.params[f"b{i}"])

            # 마지막 출력층 전까지만 BatchNorm/ReLU/Dropout을 붙인다.
            if i < len(layer_sizes) - 1:
                if use_batchnorm:
                    self.layers[f"BatchNorm{i}"] = BatchNorm(
                        self.params[f"gamma{i}"],
                        self.params[f"beta{i}"]
                    )

                # Affine 결과에 비선형성을 넣어 단순 직선 모델이 아니게 만든다.
                self.layers[f"ReLU{i}"] = ReLU()

                if use_dropout:
                    # 학습 중 일부 뉴런을 꺼서 특정 뉴런에만 의존하는 것을 줄인다.
                    self.layers[f"Dropout{i}"] = Dropout(dropout_ratio)

        # 마지막 출력 점수를 확률처럼 바꿀 Softmax 객체를 따로 둔다.
        self.softmax = Softmax()

        # raise NotImplementedError("NeuralNetwork.__init__을 구현하세요.")

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # TODO: self.layers를 순서대로 통과시키고 마지막에 Softmax를 적용하세요.
        # 처음 입력 x를 작업용 변수 out에 담고, 층을 지날 때마다 out을 새 결과로 갈아끼운다.
        out = x

        # OrderedDict에 넣어둔 순서 그대로 Affine -> BatchNorm -> ReLU -> Dropout을 통과한다.
        for layer in self.layers.values():
            # BatchNorm과 Dropout은 학습/추론 모드에 따라 동작이 달라서 train 값을 같이 넘긴다.
            if isinstance(layer, BatchNorm) or isinstance(layer, Dropout):
                out = layer.forward(out, train=train)
            else:
                # Affine과 ReLU는 입력 out만 받으면 된다.
                out = layer.forward(out)
        # 마지막 Affine 결과는 아직 점수(logit)이므로 Softmax로 확률처럼 바꾼다.
        out = self.softmax.forward(out)
        return out
    
        # raise NotImplementedError("NeuralNetwork.forward를 구현하세요.")

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # TODO: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모으세요.
        # 구현 완료: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모은다.
        # optimizer가 나중에 params와 같은 이름으로 찾아 쓸 수 있게 기울기 보관함을 새로 만든다.
        self.grads = {}

        # forward 마지막에 지나간 Softmax부터 거꾸로 통과한다.
        # 이 과제의 Softmax.backward는 이미 만들어진 기울기를 그대로 넘긴다.
        dout = self.softmax.backward(dout)

        # forward 때 쌓은 layer 순서를 뒤집어서, 마지막 layer부터 첫 layer까지 거꾸로 지나간다.
        for name, layer in reversed(list(self.layers.items())):
            # 현재 layer의 backward를 실행하고, 그 결과를 바로 앞 layer로 넘길 dout으로 갈아끼운다.
            dout = layer.backward(dout)

            # Affine 층이면 W와 b를 얼마나 수정할지 계산된 값을 grads에 저장한다.
            if isinstance(layer, Affine):
                # name이 "Affine3"이면 idx는 "3"이 된다.
                idx = name.replace("Affine", "")
                
                # params의 "W3", "b3"와 같은 이름으로 저장해야 optimizer가 짝을 맞출 수 있다.
                self.grads[f"W{idx}"] = layer.dW
                self.grads[f"b{idx}"] = layer.db

            # BatchNorm 층이면 gamma와 beta를 얼마나 수정할지 계산된 값을 grads에 저장한다.
            if isinstance(layer, BatchNorm):
                # name이 "BatchNorm2"이면 idx는 "2"가 된다.
                idx = name.replace("BatchNorm", "")
                
                # params의 "gamma2", "beta2"와 같은 이름으로 저장한다.
                self.grads[f"gamma{idx}"] = layer.dgamma
                
                self.grads[f"beta{idx}"] = layer.dbeta
                
        # raise NotImplementedError("NeuralNetwork.backward를 구현하세요.")

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)

