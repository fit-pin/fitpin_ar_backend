# 신체 측정 API
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
from src.Constant import BODY_PARTS, RES_DIR
from src.Utills import verifyValue, distance, findRealSize, reSizeofWidth

router = APIRouter()


# 신체 측정 api
@router.post("/")
async def bodyMEAApi(anaFile: UploadFile, req: Request, personKey: float = Form()):
    # 이미지인지 예외처리
    try:
        person_encoded = np.fromfile(anaFile.file, dtype=np.uint8)
        person_decode = cv.imdecode(person_encoded, cv.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=500, detail="not_image")

    try:
        # 파일명 에서 확장자 구하기
        exte = anaFile.filename.split(".")[-1]

        # uuid 로 랜덤 파일명 부여
        fileName = f"{uuid.uuid4()}.{exte}"

        # width 700 픽셀로 조정
        personImg = reSizeofWidth(person_decode, 700)
        cv.imwrite(path.join(RES_DIR, fileName), personImg)

        work = WorkBodyMEA(personKey, personImg)
        humanMEA = work.getHumanMEA()
    except Exception as e:
        print(f"애러 {req.client.host}: {e}")
        os.remove(f"{RES_DIR}/{fileName}")
        raise HTTPException(status_code=500, detail=f"{e}")

    print({"ip": req.client.host, "fileName": fileName, "result": humanMEA})

    res = {"fileName": fileName, "result": humanMEA}

    return JSONResponse(content=res, media_type="application/json")


# 신체 파트 파싱 상수 정의
PARES_TOP = {
    "왼쪽 팔": ["왼쪽 어깨", "왼쪽 팔꿈치", "왼쪽 손목"],
    "오른쪽 팔": ["오른쪽 어깨", "오른쪽 팔꿈치", "오른쪽 손목"],
    "어께너비": ["왼쪽 어깨", "오른쪽 어깨"],
    "상체너비": ["왼쪽 어깨", "왼쪽 골반"],
}
PARES_BOTTOM = {
    "왼쪽 다리": ["왼쪽 골반", "왼쪽 무릎", "왼쪽 발"],
    "오른쪽 다리": ["오른쪽 골반", "오른쪽 무릎", "오른쪽 발"],
}


class WorkBodyMEA:
    model = YOLO("src/model/yolov8n-pose.pt")

    def __init__(self, personKey: int, img: cv.typing.MatLike):
        self.personKey = personKey
        self.img = img

    def getHumanMEA(self) -> dict[Literal["armSize", "shoulderSize", "bodySize", "legSize"], float]:
        """입력된 정보로 신체 측정

        Returns:
            dict: [armSize]: 팔 길이 [shoulderSize]: 어께 너비 [bodySize]: 상체 길이 [legSize]: 다리 길이

        Raise:
            not_detection: 사람 감지 안됨
            many_detection: 여러 사람 감지됨
            keypoint_err: 키포인트 검출실패
        """

        result: Results = self.model.predict(self.img)[0]
        # 사람 감지 인덱스
        detcIndex = -1

        for i in range(len(result.boxes.cls)):
            if result.boxes.cls[i] == 0:
                if detcIndex != -1:
                    # 두명 이상 감지되면 예외
                    raise Exception("many_detection")
                detcIndex = i

        # 사람 감지 안되면 예외
        if detcIndex == -1:
            raise Exception("not_detection")

        # 사람 키포인트
        personPose = result.keypoints.xy[detcIndex]
        # 사람 영역
        personPx = float(result.boxes[detcIndex].xywh[0][3])

        # 사람 키포인트 유효성 검사
        def __findPoints(key: str):
            point = personPose[BODY_PARTS[key]]
            if not verifyValue(point):
                raise Exception("keypoint_err")
            return point

        shoulder = list(map(__findPoints, PARES_TOP["어께너비"]))
        body = list(map(__findPoints, PARES_TOP["상체너비"]))

        rightArm = list(map(__findPoints, PARES_TOP["오른쪽 팔"]))
        leftArm = list(map(__findPoints, PARES_TOP["왼쪽 팔"]))

        rightLeg = list(map(__findPoints, PARES_BOTTOM["오른쪽 다리"]))
        leftLeg = list(map(__findPoints, PARES_BOTTOM["왼쪽 다리"]))

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

        return {"armSize": armSize, "shoulderSize": shoulderSize, "bodySize": bodySize, "legSize": legSize}
