# 신체 측정 API
import logging
import os
from os import path
from typing import Literal
from fastapi import APIRouter, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request
import uuid
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

import Constant
from Utills import verifyValue, distance, findRealSize, reSizeofWidth
from rembg import remove, new_session

router = APIRouter()
logger = logging.getLogger('uvicorn.error')

# 신체 측정 api
@router.post("/")
async def bodyMEAApi(anaFile: UploadFile, req: Request, personKey: float = Form()):
    # 이미지인지 예외처리
    try:
        person_encoded = np.fromfile(anaFile.file, dtype=np.uint8)
        person_decode = cv.imdecode(person_encoded, cv.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=500, detail="not_image")

    # 파일명 에서 확장자 구하기
    exte = anaFile.filename.split(".")[-1] # type: ignore

    # uuid 로 랜덤 파일명 부여
    fileName = f"{uuid.uuid4()}.{exte}"

    try:
        # width 700 픽셀로 조정
        personImg = reSizeofWidth(person_decode, 700)

        work = WorkBodyMEA(personKey, personImg)
        humanMEA = work.getHumanMEA()

        # 체형 이미지 저장
        work.saveHumanImage(path.join(Constant.RES_DIR, fileName))
    except Exception as e:
        logger.error(f"{req.client.host}:{req.client.port} - 애러: {e}") # type: ignore
        if path.exists(path.join(Constant.RES_DIR, fileName)):
            os.remove(path.join(Constant.RES_DIR, fileName))
        raise HTTPException(status_code=500, detail=f"{e}")

    res = {"fileName": fileName, "result": humanMEA}
    logger.info(f"{req.client.host}:{req.client.port} - 채형측정 완료: {res}") # type: ignore

    return JSONResponse(content=res, media_type="application/json")


class WorkBodyMEA:
    def __init__(self, personKey: float, img: cv.typing.MatLike):
        self.personKey = personKey
        self.img = img

    def getHumanNukki(self) -> cv.typing.MatLike:
        """체형이미지에 사람을 누끼딴 이미지를 반환합니다.

        Args:
            fileName (str): 저장할 파일명
        """
        session = new_session("u2net_human_seg")

        result_img = remove(
            self.img,
            alpha_matting=True,
            alpha_matting_foreground_threshold=20,
            alpha_matting_background_threshold=1,
            alpha_matting_erode_size=1,
            bgcolor=(255, 255, 255, 255),
            session=session,
        )
        
        return result_img # type: ignore
    
    
    def saveHumanImage(self, fileName: str):
        """채형 사진을 저장합니다

        Args:
            fileName (str): 저장경로
        """
        cv.imwrite(fileName, self.img)
        
    
    def getHumanMEA(
        self,
    ) -> dict[Literal["armSize", "shoulderSize", "bodySize", "legSize"], float]:
        """입력된 정보로 신체 측정

        Returns:
            dict: [armSize]: 팔 길이 [shoulderSize]: 어께 너비 [bodySize]: 상체 길이 [legSize]: 다리 길이

        Raise:
            not_detection: 사람 감지 안됨
            many_detection: 여러 사람 감지됨
            keypoint_err: 키포인트 검출실패
        """

        result: Results = WorkBodyMEA.MODEL.predict(self.img)[0]
        # 사람 감지 인덱스
        detcIndex = -1

        for i in range(len(result.boxes.cls)): # type: ignore
            if result.boxes.cls[i] == 0: # type: ignore
                if detcIndex != -1:
                    # 두명 이상 감지되면 예외
                    raise Exception("many_detection")
                detcIndex = i

        # 사람 감지 안되면 예외
        if detcIndex == -1:
            raise Exception("not_detection")

        # 사람 키포인트
        personPose = result.keypoints.xy[detcIndex] # type: ignore
        # 사람 영역
        personPx = float(result.boxes[detcIndex].xywh[0][3]) # type: ignore

        # 사람 키포인트 유효성 검사
        def __findPoints(key: str):
            point = personPose[WorkBodyMEA.BODY_PARTS[key]]
            if not verifyValue(point):
                raise Exception("keypoint_err")
            return point

        shoulder = list(map(__findPoints, WorkBodyMEA.PARES_TOP["어께너비"]))
        body = list(map(__findPoints, WorkBodyMEA.PARES_TOP["상체너비"]))

        rightArm = list(map(__findPoints, WorkBodyMEA.PARES_TOP["오른쪽 팔"]))
        leftArm = list(map(__findPoints, WorkBodyMEA.PARES_TOP["왼쪽 팔"]))

        rightLeg = list(map(__findPoints, WorkBodyMEA.PARES_BOTTOM["오른쪽 다리"]))
        leftLeg = list(map(__findPoints, WorkBodyMEA.PARES_BOTTOM["왼쪽 다리"]))

        shoulderSize = findRealSize(self.personKey, personPx, distance(shoulder))
        shoulderSize = round(shoulderSize, 2)

        bodySize = findRealSize(self.personKey, personPx, distance(body))
        bodySize = round(bodySize, 2)

        armSize = max(
            [
                findRealSize(self.personKey, personPx, distance(rightArm)),
                findRealSize(self.personKey, personPx, distance(leftArm)),
            ]
        )
        armSize = round(armSize, 2)

        legSize = max(
            [
                findRealSize(self.personKey, personPx, distance(rightLeg)),
                findRealSize(self.personKey, personPx, distance(leftLeg)),
            ]
        )
        legSize = round(legSize, 2)

        return {
            "armSize": armSize,
            "shoulderSize": shoulderSize,
            "bodySize": bodySize,
            "legSize": legSize,
        }
        
    # 상수 정의
    BODY_PARTS = {
        "코": 0,
        "오른쪽 눈": 1,
        "왼쪽 눈": 2,
        "오른쪽 귀": 3,
        "왼쪽 귀": 4,
        "오른쪽 어깨": 5,
        "왼쪽 어깨": 6,
        "오른쪽 팔꿈치": 7,
        "왼쪽 팔꿈치": 8,
        "오른쪽 손목": 9,
        "왼쪽 손목": 10,
        "오른쪽 골반": 11,
        "왼쪽 골반": 12,
        "오른쪽 무릎": 13,
        "왼쪽 무릎": 14,
        "오른쪽 발": 15,
        "왼쪽 발": 16,
    }
    """신체 파트 구분"""
            

    PARES_TOP = {
        "왼쪽 팔": ["왼쪽 어깨", "왼쪽 팔꿈치", "왼쪽 손목"],
        "오른쪽 팔": ["오른쪽 어깨", "오른쪽 팔꿈치", "오른쪽 손목"],
        "어께너비": ["왼쪽 어깨", "오른쪽 어깨"],
        "상체너비": ["왼쪽 어깨", "왼쪽 골반"],
    }
    """상체 파트 파싱 상수"""
    
    PARES_BOTTOM = {
        "왼쪽 다리": ["왼쪽 골반", "왼쪽 무릎", "왼쪽 발"],
        "오른쪽 다리": ["오른쪽 골반", "오른쪽 무릎", "오른쪽 발"],
    }
    """하체 파트 파싱 상수"""
    
    MODEL = YOLO("src/model/yolov8n-pose.pt")
    """POSE 모델"""
