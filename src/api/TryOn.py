# 의류 합성 이미지 생성 API
from typing import Literal
from torch import Tensor
from ultralytics.engine.results import Results
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
import cv2 as cv
import cvzone
import numpy as np
from os import path

from ultralytics import YOLO

from src.Constant import BODY_PARTS, RES_DIR
from src.Utills import distance, reSizeofWidth

router = APIRouter()


@router.post("/")
def tryOn(
    clothesImg: UploadFile,
    req: Request,
    clothesType: Literal["TOP", "BOTTOM"] = Form(),
    fileName: str = Form(),
    personKey: float = Form(),
    clothesLenth: float = Form(),
):
    bodyPath = path.join(RES_DIR, fileName)
    try:
        if not path.exists(bodyPath):
            raise Exception("not_exists_bodyImg")

        clothes_encoded = np.fromfile(clothesImg.file, dtype=np.uint8)
        clothes_decode = cv.imdecode(clothes_encoded, cv.IMREAD_UNCHANGED)

        personImg = cv.imread(bodyPath)

        workTryOn = WorkTryOn(personImg, clothes_decode, clothesType)
    except Exception as e:
        print(f"애러 {req.client.host}: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")

    return {"res": "clothesMEA"}


class WorkTryOn:
    model = YOLO("src/model/yolov8n-pose.pt")

    def __init__(
        self, personImg: cv.typing.MatLike, clothesImg: cv.typing.MatLike, clothesType: Literal["TOP", "BOTTOM"]
    ):
        """TryOn 작업 생성하기

        Args:
            personImg (cv.typing.MatLike): 채형 이미지
            clothesImg (cv.typing.MatLike): 의류 이미지 (누끼 따진)
            clothesType (Literal["TOP", "BOTTOM"]): 상의 하의 구분
        """
        self.personImg = personImg
        self.clothesImg = clothesImg
        self.clothesType: Literal["TOP", "BOTTOM"] = clothesType