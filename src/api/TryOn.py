# 의류 합성 이미지 생성 API
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
    fileName: str = Form(),
    personKey: float = Form(),
    clothesLenth: int = Form()
):
    bodyPath = path.join(RES_DIR, fileName)
    try:
        if not path.exists(bodyPath):
            raise Exception("not_exists_bodyImg")

        clothes_encoded = np.fromfile(clothesImg.file, dtype=np.uint8)
        clothes_decode = cv.imdecode(clothes_encoded, cv.IMREAD_UNCHANGED)

        personImg = cv.imread(bodyPath)

        workTryOn = WorkTryOn(personImg, clothes_decode)
        resultsImg = workTryOn.getTryOnImg()
    except Exception as e:
        print(f"애러 {req.client.host}: {e}")
        raise HTTPException(status_code=500, detail=f"{e}")

    return Response(cv.imencode(".png", resultsImg)[1].tobytes(), media_type="image/png")


class WorkTryOn:
    model = YOLO("src/model/yolov8n-pose.pt")

    def __init__(self, personImg: cv.typing.MatLike, clothesImg: cv.typing.MatLike):
        """TryOn 작업 생성하기

        Args:
            personImg (cv.typing.MatLike): 채형 이미지
            clothesImg (cv.typing.MatLike): 의류 이미지 (누끼 따진)
        """
        self.personImg = personImg
        self.clothesImg = clothesImg

    def __overlayClothes(self, personPose: Tensor) -> cv.typing.MatLike:
        # 의류 이미지 가로 실제 보정 배율
        WIDTH_CORR = 2.1

        # X 좌표 보정 배율
        X_POINT_CORR = 0.79

        # Y 좌표 보정 배율
        Y_POINT_CORR = 0.93

        # 상채 길이를 기준으로
        point1 = personPose[BODY_PARTS["왼쪽 어깨"]]
        point2 = personPose[BODY_PARTS["오른쪽 어깨"]]

        # 어깨와 어꺠 사이로 이미지 사이즈 보정 값 얻기
        findDistance = distance([point1, point2]) * WIDTH_CORR

        # 이미지 크기 보정하기
        resize_clothes = reSizeofWidth(self.clothesImg, int(findDistance))

        # 이미지가 합성될 위치 지정
        x, y = point1

        # 사람 이미지 투명도 값 추가
        personimg_bgra = cv.cvtColor(self.personImg, cv.COLOR_BGR2BGRA)

        # 체형 이미지와 보정된 의류 이미지 합성
        return cvzone.overlayPNG(personimg_bgra, resize_clothes, (int(x * X_POINT_CORR), int(y * Y_POINT_CORR)))

    def getTryOnImg(self) -> cv.typing.MatLike:
        """채형사진과 의류 사진이 합생된 이미자를 리턴 합니다

        Raises:
            not_detection: 사람 감지 안됨
            many_detection: 여러 사람 감지됨
            keypoint_err: 키포인트 검출실패

        Returns:
            MatLike: 합성된 이미지
        """
        result: Results = self.model.predict(self.personImg)[0]
        # 사람 감지 인덱스
        detcIndex = -1

        for i in range(len(result.boxes.cls)):
            print(result.boxes.cls)
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

        # 신체 이미지와 의류 이미지 합성
        return self.__overlayClothes(personPose)
