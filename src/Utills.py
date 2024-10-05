import math
from typing import Literal

import cv2 as cv
import numpy as np
from torch import Tensor
import torchvision.transforms as transforms

Normal = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
"""이미지 정규화 구성"""


def findRealSize(refSize: float, refPx: float, findPx: float):
    """기준 사물 높이 가지고 다른 사이즈 예측

    Args:
        refSize (float): 기준 사물 크기(cm)
        refPx (float): 기준 사물 픽셀상 크기
        findPx (float): 찾으려는 사물 픽셀상 크기

    Returns:
        float: 사물사이즈(cm)
    """

    cm_per_px = refSize / refPx
    return findPx * cm_per_px


def distance(points: Tensor | list) -> float:
    """여러 점들 사이 길이 구하는 함수

    Args:
        points (list[tuple[float]]): (x,  y) 이걸 구하고 싶은 만큼 list로 묶어서

    Returns:
        float: 점 전체 길이
    """

    distance = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def reSizeofWidth(img: cv.typing.MatLike, reWidth: int):
    """width 값 기준으로 사진을 줄임 height 값은 비율에따라 자동 조정

    Args:
        img (cv.typing.MatLike): openCV 이미지 파일
        reWidth (int): 줄이는 width값

    Returns:
        cv.typing.MatLike: 완료된 openCV 이미지 파일
    """

    height, width = img.shape[:2]
    reHeight = int(height * reWidth / width)
    return cv.resize(img, (reWidth, reHeight), interpolation=cv.INTER_AREA)


def reSizeofHight(img: cv.typing.MatLike, reHight: int):
    """hight 값 기준으로 사진을 줄임 width 값은 비율에따라 자동 조정

    Args:
        img (cv.typing.MatLike): openCV 이미지 파일
        reHight (int): 줄이는 hight값

    Returns:
        cv.typing.MatLike: 완료된 openCV 이미지 파일
    """

    height, width = img.shape[:2]
    reWidth = int(width * reHight / height)
    return cv.resize(img, (reWidth, reHight), interpolation=cv.INTER_AREA)


def verifyValue(values: Tensor | list):
    """리스트 값 유효성 확인

    Args:
        values (list[float  |  Tensor]): 평가할 값

    Returns:
        bool: 유효성 여부
    """

    for value in values:
        if value == 0:
            return False
    return True


def getNormalizimage(img: cv.typing.MatLike):
    """
    이미지를 정규화 하여 반환 합니다.</br>
    반환된 이미지로 `getKeyPointsResult()` 함수를 호출하여 keyPoint 를 예측합니다

    Args:
        img (cv2.typing.MatLike): cv 이미지

    Returns:
        Tensor: 정규화된 이미지 텐서
    """
    nom = Normal(img)
    res = Tensor(np.expand_dims(nom, axis=0))
    return res


def resizeWithPad(
    image: cv.typing.MatLike,
    new_shape: tuple[int, int],
    padding_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[cv.typing.MatLike, dict[Literal["top", "bottom", "left", "right"], int]]:
    """
    비율을 유지하여 이미지를 자릅니다.</br>
    이때 비율을 유지하기 위해 잘려진 부분은 `padding_color`로 채워 집니다

    Params:
        image (MatLike): 원본 이미지
        new_shape (Tuple[int, int]): 바꿀 크기
        padding_color (Tuple[int, int, int]): 잘려진 부분 색상
    Returns:
        tuple[cv2.typing.MatLike, dict[str, int]]</br>
        - `[0]`: 잘려진 이미지
        - `[1]`: paddig 값
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])

    test_w = new_shape[0] - new_size[0]
    test_h = new_shape[1] - new_size[1]

    if test_w < 0:
        ratio = float(new_shape[0] / new_size[0])
        new_size = tuple([int(x * ratio) for x in new_size])
    elif test_h < 0:
        ratio = float(new_shape[1] / new_size[1])
        new_size = tuple([int(x * ratio) for x in new_size])

    image = cv.resize(image, new_size)

    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv.copyMakeBorder(
        image, top, bottom, left, right, cv.BORDER_CONSTANT, value=padding_color
    )

    return image, {"top": top, "bottom": bottom, "left": left, "right": right}
