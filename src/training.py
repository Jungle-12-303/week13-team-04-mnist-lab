# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # 구현 완료: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행한다.
    # epoch별 평균 loss를 보고 학습이 내려가는지 확인하기 위한 기록장이다.
    loss_history = []
    # 전체 학습 데이터 개수다. 이 숫자를 기준으로 매 epoch마다 index를 섞는다.
    num_train = x_train.shape[0]

    for epoch in range(epochs):
        # 원본 데이터를 직접 섞지 않고, 번호표만 랜덤 순서로 섞는다.
        indices = np.random.permutation(num_train)
        # 이번 epoch의 batch loss를 모두 더해 평균을 내기 위한 누적값이다.
        epoch_loss = 0
        # 이번 epoch에서 mini-batch가 몇 번 돌았는지 센다.
        batch_count = 0

        for start in range(0, num_train, batch_size):
            # start부터 batch_size만큼 잘라 현재 batch 범위를 만든다.
            end = start + batch_size
            batch_indices = indices[start:end]

            # 섞인 번호표를 이용해 실제 입력 이미지와 정답 label batch를 꺼낸다.
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            # 1단계 Forward: 현재 batch를 모델에 넣어 숫자별 확률을 만든다.
            y_pred = model.forward(x_batch, train=True)
            # 2단계 Loss: 예측 확률과 정답 label을 비교해 벌점 값을 계산한다.
            loss = cross_entropy_loss(y_pred, y_batch)

            # 마지막 batch는 batch_size보다 작을 수 있으므로 실제 batch 크기를 따로 본다.
            batch_size_actual = x_batch.shape[0]
            # softmax + cross entropy를 합친 미분 시작값을 만든다.
            dout = y_pred.copy()
            # 각 데이터 줄에서 정답 칸만 1을 빼면, 예측과 정답의 차이가 gradient가 된다.
            dout[np.arange(batch_size_actual), y_batch] -= 1
            # batch 평균 loss였으므로 gradient도 batch 개수로 나눠 평균으로 맞춘다.
            dout /= batch_size_actual

            # 3단계 Backward: 마지막 출력층에서 첫 layer까지 거꾸로 gradient를 흘린다.
            model.backward(dout)
            # 4단계 Update: optimizer가 params와 grads를 보고 실제 W/b/gamma/beta 값을 고친다.
            optimizer.update(model.params, model.grads)

            # 이번 batch의 loss를 epoch 평균 계산용으로 모은다.
            epoch_loss += loss
            batch_count += 1

        # epoch 하나가 끝나면 batch loss 평균을 history에 저장한다.
        loss_history.append(epoch_loss / batch_count)

    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
