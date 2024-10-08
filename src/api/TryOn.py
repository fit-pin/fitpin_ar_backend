# 의류 합성 이미지 생성 API
import logging
from os import path
from typing import Literal

from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile

import Constant
from requests import post as reqPost
from requests_toolbelt.multipart.encoder import MultipartEncoder

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.post("/")
def tryOn(
    req: Request,
    clothesImg: UploadFile,
    bodyFileName: str = Form(),
    category: Literal["상의", "하의", "드레스"] = Form(),
    is_checked: bool = Form(True),
    is_checked_crop: bool = Form(True),
    denoise_steps: int = Form(30),
    seed: int = Form(42),
):
    try:
        logger.info(f"{req.client.host}:{req.client.port} - AR_TryOn 진행중")  # type: ignore

        bodyPath = path.join(Constant.RES_DIR, bodyFileName)
        if not path.exists(bodyPath):
            raise Exception("not_exists_bodyImg")

        # multipart-form body 생성
        mutiPartBody = MultipartEncoder(
            fields={
                # 파일 전송
                "humanImg": (
                    bodyFileName,
                    open(bodyPath, "rb").read(),
                    "image/jpeg",
                ),
                "clothesImg": (
                    clothesImg.filename,
                    clothesImg.file.read(),
                    clothesImg.content_type,
                ),
                "category": category,
                "is_checked": str(is_checked),
                "is_checked_crop": str(is_checked_crop),
                "denoise_steps": str(denoise_steps),
                "seed": str(seed),
            }
        )

        res = reqPost(
            Constant.IDM_URL,
            data=mutiPartBody,
            headers={"Content-Type": mutiPartBody.content_type},
        )

        logger.info(f"{req.client.host}:{req.client.port} - IDM-VTON 서버에 정상적으로 요청 전달")  # type: ignore

        if not res.ok:
            raise Exception(res.json())
        logger.info(f"{req.client.host}:{req.client.port} - AR_TryOn 성공적으로 응답")  # type: ignore
        return Response(res._content, media_type=res.headers["content-type"])
    except Exception as e:
        logger.error(f"{req.client.host}:{req.client.port} - 애러: {e}")  # type: ignore
        raise HTTPException(status_code=500, detail=f"{e}")
