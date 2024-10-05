# 누끼 따는 API
import logging
from fastapi import APIRouter
import cv2 as cv
import numpy as np
from rembg import remove

router = APIRouter()
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response

logger = logging.getLogger('uvicorn.error')

@router.post("/")
def getNukki(clothesImg: UploadFile, req: Request):
    try:
        clothes_encoded = np.fromfile(clothesImg.file, dtype=np.uint8)
        clothes_decode = cv.imdecode(clothes_encoded, cv.COLOR_BGR2RGB)
        workgetNukki = WorkgetNukki(clothes_decode)
        resultsImg = workgetNukki.getNukkImg()
    except Exception as e:
        logger.error(f"{req.client.host}:{req.client.port} - 애러: {e}") # type: ignore
        raise HTTPException(status_code=500, detail=f"{e}")

    return Response(
        cv.imencode(".png", resultsImg)[1].tobytes(), media_type="image/png"
    )


class WorkgetNukki:
    def __init__(self, clothesImg: cv.typing.MatLike):
        self.clothesImg = clothesImg

    def getNukkImg(self) -> cv.typing.MatLike:
        """
        해당 의류의 누끼를 제거합니다.</br>
        이때 배경 영역인 부분은 의류 이미지와 딱 맞게 재조정 됩니다.

        Returns:
            cv.typing.MatLike: 누끼 따진 이미지
        """

        # 누끼 따기
        result_img: cv.typing.MatLike = remove(
            self.clothesImg,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        ) # type: ignore

        # 알파 채널 분리
        alpha_channel = result_img[:, :, 3]

        # 투명하지 않은 영역 찾기
        coords = cv.findNonZero(alpha_channel)

        # 투명하지 않은 영역의 바운딩 박스 계산
        x, y, w, h = cv.boundingRect(coords)

        # 바운딩 박스를 기준으로 이미지 자르기
        cropped_image = result_img[y : y + h, x : x + w]

        return cropped_image
