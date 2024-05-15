import math

import cv2 as cv
from torch import Tensor

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

def distance(points: list[tuple[float]]) -> float:
    """여러 점들 사이 길이 구하는 함수

    Args:
        points (list[tuple[float]]): (x,  y) 이걸 구하고 싶은 만큼 list로 묶어서

    Returns:
        float: 점 전체 길이
    """

    distance = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def reSizeofWidth(img: cv.typing.MatLike, width: int):
    """width 값 기준으로 사진을 줄임 height 값은 비율에따라 자동 조정

    Args:
        img (cv.typing.MatLike): openCV 이미지 파일
        width (int): 줄이는 width값

    Returns:
        cv.typing.MatLike: 완료된 openCV 이미지 파일
    """

    height, width = img.shape[:2]
    reHeight = int(height * width / width)
    return cv.resize(img, (width, reHeight), interpolation=cv.INTER_AREA)

def reSizeofHight(img: cv.typing.MatLike, hight: int):
    """hight 값 기준으로 사진을 줄임 width 값은 비율에따라 자동 조정

    Args:
        img (cv.typing.MatLike): openCV 이미지 파일
        hight (int): 줄이는 hight값

    Returns:
        cv.typing.MatLike: 완료된 openCV 이미지 파일
    """

    height, width = img.shape[:2]
    reWidth = int(height * width / height)
    return cv.resize(img, (reWidth, height), interpolation=cv.INTER_AREA)

def verifyValue(values: list[float | Tensor]):
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
