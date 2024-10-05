# 의류 합성 이미지 생성 API
import logging
from typing import Literal
import uuid
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from os import path, mkdir
from shutil import rmtree

from gradio_client import Client, handle_file

from src.Constant import RES_DIR

router = APIRouter()
logger = logging.getLogger('uvicorn.error')

gradioClient = Client("yisol/IDM-VTON", download_files=path.join(RES_DIR, "temp"))

@router.post("/")
def tryOn(
    clothesImg: UploadFile,
    req: Request,
    clothesType: Literal["TOP", "BOTTOM"] = Form(),
    fileName: str = Form(),
):
    # temp 폴더 비우기
    if path.exists(path.join(RES_DIR, "temp")):
        rmtree(path.join(RES_DIR, "temp"))

    mkdir(path.join(RES_DIR, "temp"))

    bodyPath = path.join(RES_DIR, fileName)
    try:
        if not path.exists(bodyPath) or not clothesImg.filename:
            raise Exception("not_exists_bodyImg")

        # 파일명 에서 확장자 구하기
        exte = clothesImg.filename.split(".")[-1]

        clothesPath = path.join(RES_DIR, "temp", f"{uuid.uuid4()}.{exte}")

        with open(clothesPath, "wb") as f:
            f.writelines(clothesImg.file)

        workTryOn = WorkTryOn(bodyPath, clothesPath, clothesType)

        logger.info(f"{req.client.host}:{req.client.port} - AR_TryOn 진행중") # type: ignore
        tryOnPath = workTryOn.getTryOnImg()
    except Exception as e:
        logger.error(f"{req.client.host}:{req.client.port} - 애러: {e}") # type: ignore
        raise HTTPException(status_code=500, detail=f"{e}")

    return FileResponse(tryOnPath, media_type="image/png")


class WorkTryOn:

    def __init__(
        self, bodyPath: str, clothesPath: str, clothesType: Literal["TOP", "BOTTOM"]
    ):
        """TryOn 작업 생성하기

        Args:
            bodyPath (cv.typing.MatLike): 채형 이미지 경로
            clothesPath (cv.typing.MatLike): 의류 이미지 경로
            clothesType (Literal["TOP", "BOTTOM"]): 상의 하의 구분
        """
        self.bodyPath = bodyPath
        self.clothesPath = clothesPath
        self.clothesType: Literal["TOP", "BOTTOM"] = clothesType

    def getTryOnImg(self):
        """
        IDM-VTON 으로 예측한 이미지를 반환합니다

        Returns:
            str: 예측이미지
        """

        result: tuple[str] = gradioClient.predict(
            dict={
                "background": handle_file(self.bodyPath),
                "layers": [],
                "composite": None,
            },
            garm_img=handle_file(self.clothesPath),
            garment_des="any",
            is_checked=True,
            is_checked_crop=True,
            denoise_steps=30,
            seed=42,
            api_name="/tryon",
        )

        return result[0]
