# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np

# def __init__():

def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).: 교차 엔트로피 오차 
    y_pred: (batch_size, 10) 확률 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    y_true: (batch_size,) 정수 레이블 0~9 = y_true = np.array([2, 0])
    -> 원 - 핫 인코딩 형태가 아닌 단일 숫자 레이블 형태 
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.

    # 정답 레이블에 해당하는 신경망의 출력값만 신경쓰면 됨
    # 항상 -log(정답 레이블 신경망 출력값) = CEE 값 
    # -> 정답 데이터 tk 가 원-핫 인코딩 형태라, 정답 인덱스만 1이고 나머지 다른 인덱스는 0이기 때문 
    # -> 오답인 경우 tk가 0 => 0 *logyk = 0 => tk가 1인 정답 레이블일 때의 logyk값 하나만 살아남음
    # -> 수식에 시그마 있어서 원래 0~9 모든 클래스에 대해 값을 계산해서 더해야 하지만, 오답인 경우 전부 0이 되어버리기 때문에 그냥 정답 레이블만 계산해서 더해주면 됨
    # np_clip: 배열 안의 값들을 지정한 최소값과 최대값 사이의 범위로 제한(자르기, Clipping)하는 함수
    # 배열의 요소 중 최소값보다 작은 값은 모두 최소값으로 변환
    # 배열의 요소 중 최대값보다 큰 값은 모두 최대값으로 변환 
    # -> 최소값과 최대값 범위 안에 있는 값들은 모두 그대로 유지  
    # np.clip으로 log(0)을 피한다: np_clip
    # np_clip(배열, 최소, 최대)

    # out = -np.sum(y_true * np.log(cliped_y)) -> ValueError: operands could not be broadcast together with shapes (2,) (2,3) 발생
    # -> 소프트맥스 함수가 반환하는 넘파이 배열은 1차원이 될 수도, 2차원이 될 수도 있음
    # => 단일 데이터에 대한 로짓(점수, score)을 입력하면 1차원 배열 반환(해당 프로젝트는 모두 미니 배치 방식으로 사용되기 때문에 1차원 배열 반환에 대응할 필요 없음)
    # => 미니 배치 방식을 사용할 경우, 입력 데이터가 행렬(2차원 배열) 형태 => 소프트맥스 함수 반환도 동일하게 2차원 배열이 됨
    # 배치 데이터를 지원하는 CEE가 필요함

    # y_pred.ndim: 넘파이 배열 y_pred의 차원 수 반환
    # -> y_pred 배열이 1차원 배열일 경우 처리
    if y_pred.ndim == 1: # -> y_pred: (10, ) 1차원 배열
            # y_pred과 y_true와 모두 배치 크기가 1인 2차원 배열로 변환 
            # reshape(-1, 1) 하면 안 됨! -> 이렇게 하면 (10, 1) 형태의 행렬이 되어서 배치 사이즈가 10개가 됨
            # 미니 배치 처리를 위해 행렬은 항상 (데이터 개수, 클래스 수) 형태가 되어야 함!
            y_pred = y_pred.reshape(1, y_pred.size)
            y_true = y_true.reshape(1, y_true.size)

    # np.log(0) -> 마이너스 무한대를 뜻하는 -inf가 됨
    # -> np.log의 입력값이 절대 0이 되지 않도록 delta(1e-7) 값을 더함
    # => np.clip을 써서 최소값을 1e-7로 해서 1e-7보다 작은 값은 전부 a + 1e-7로 바꾸라는 것
    # => 절대 y + 1e-7로 하면 안 됨 -> 정상적인 값까지 무조건 전부 + 1e-7 처리 되어서 값이 오염됨 
    # 신경망 출력은 소프트맥스 함수 값이 최대 1.0이니까 최대값을 1.0으로 해서 처리
    cliped_y = np.clip(y_pred, 1e-7, 1.0)

    # np.log(cliped_y): logyk 연산
    # CEE 시그마 수식대로 tk * logyk 계산 후 모든 값들을 더하고 앞에 -를 붙여줘야 함

    # y_pred의 형태를 가져와 행 개수를 확인 -> 데이터 개수를 확인
    # 평균을 구하기 위한 배치의 크기로 설정 
    batch_size = y_pred.shape[0]

    # 배치의 크기로 나눠 정규화 -> 이미지 1장당 평균의 CEE 계산 
    # 미니 배치 CEE 수식 적용
    # out = -(np.sum(y_true * np.log(cliped_y)) / batch_size) -> 정답 레이블이 원-핫 인코딩 형태일 때의 수식

    # 정답 레이블이 원-핫 인코딩 형태가 아니기 때문에 np.arrange 사용해서 0부터 batch_size - 1 까지 배열을 생성하기
    # -> 각 데이터의 정답 레이블에 해당하는 신경망 출력을 추출함 
    out = -(np.sum(np.log(cliped_y[np.arange(batch_size), y_true])) / batch_size)

    return out
    # raise NotImplementedError("cross_entropy_loss를 구현하세요.")
