# 의류 합성 이미지 생성 API
import logging
from os import path, remove
from shutil import rmtree
from typing import Literal
import uuid

from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile
from gradio_client import Client, handle_file

import Constant
from requests import post as reqPost
from requests_toolbelt.multipart.encoder import MultipartEncoder

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

if Constant.TRYON_MODE == "huggingface":
    gradioClient = Client(
        Constant.TRYON["huggingface"],
        download_files=path.join(Constant.RES_DIR, "temp"),
    )


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

        workTryOn = WorkTryOn(
            req,
            clothesImg,
            bodyFileName,
            bodyPath,
            category,
            is_checked,
            is_checked_crop,
            denoise_steps,
            seed,
        )

        result: bytes
        if Constant.TRYON_MODE == "huggingface":
            logger.info("TryOn - HuggingFace API를 사용하여 예측")
            result = workTryOn.tryOnHuggingface()
        else:
            logger.info("TryOn - 자체 구현 FastAPI 서버를 사용하여 예측")
            result = workTryOn.tryOnLocal()

        logger.info(f"{req.client.host}:{req.client.port} - AR_TryOn 성공적으로 응답")  # type: ignore
        return Response(result, media_type=clothesImg.content_type)
    except Exception as e:
        logger.error(f"{req.client.host}:{req.client.port} - TryOn 애러: {e}")  # type: ignore
        raise HTTPException(status_code=500, detail=f"{e}")


class WorkTryOn:
    def __init__(
        self,
        req: Request,
        clothesImg: UploadFile,
        bodyFileName: str,
        bodyPath: str,
        category: Literal["상의", "하의", "드레스"],
        is_checked: bool,
        is_checked_crop: bool,
        denoise_steps: int,
        seed: int,
    ):
        """TryOn 작업 생성하기

        Args:
            req (Request): Request 객체
            clothesImg (UploadFile): 의류사진 (바이너리)
            bodyFileName (str): AR 서버에 저장된 체형파일 명
            bodyPath (str): AR 서버에 저장된 체형파일 경로
            category (Literal[): 의류 종류
            is_checked (bool): Use auto-generated mask 설정
            is_checked_crop (bool): 크롭 사용
            denoise_steps (int): 노이즈 재거 단계
            seed (int): 랜덤시드
        """

        self.req = req
        self.clothesImg = clothesImg
        self.bodyFileName = bodyFileName
        self.bodyPath = bodyPath
        self.category = category
        self.is_checked = is_checked
        self.is_checked_crop = is_checked_crop
        self.denoise_steps = denoise_steps
        self.seed = seed

    def tryOnHuggingface(self) -> bytes:
        """
        Huggingface api 로 IDM-VTON 을 사용

        Returns:
            bytes: 가상 피팅 의류 이미지 바이트 바이너리
        """

        # 파일명 에서 확장자 구하기
        exte = self.clothesImg.filename.split(".")[-1]  # type: ignore

        clothesPath = path.join(Constant.RES_DIR, "temp", f"{uuid.uuid4()}.{exte}")

        # 가상 피팅을 위해 임시 저장
        with open(clothesPath, "wb") as f:
            f.writelines(self.clothesImg.file)

        result: tuple[str, str] = gradioClient.predict(
            dict={
                "background": handle_file(self.bodyPath),
                "layers": [],
                "composite": None,
            },
            garm_img=handle_file(clothesPath),
            garment_des="any",
            is_checked=self.is_checked,
            is_checked_crop=self.is_checked_crop,
            denoise_steps=self.denoise_steps,
            seed=self.seed,
            api_name="/tryon",
        )

        # 저장된 피팅 사진 가져 오기
        image = open(result[0], "rb").read()

        logger.info(f"{self.req.client.host}:{self.req.client.port} - tryOnHuggingface 성공적으로 예측")  # type: ignore

        # 저장된 임시 파일들 삭제
        rmtree(path.dirname(result[0]))
        rmtree(path.dirname(result[1]))
        remove(clothesPath)

        return image

    def tryOnLocal(self) -> bytes:
        """
        자체 구현한 Fast-API 서버로 IDM-VTON 을 사용


        Raises:
            Exception: 서버 오류

        Returns:
            bytes: 가상 피팅 의류 이미지 바이트 바이너리
        """

        # multipart-form body 생성
        mutiPartBody = MultipartEncoder(
            fields={
                # 파일 전송
                "humanImg": (
                    self.bodyFileName,
                    open(self.bodyPath, "rb").read(),
                    "image/jpeg",
                ),
                "clothesImg": (
                    self.clothesImg.filename,
                    self.clothesImg.file.read(),
                    self.clothesImg.content_type,
                ),
                "category": self.category,
                "is_checked": str(self.is_checked),
                "is_checked_crop": str(self.is_checked_crop),
                "denoise_steps": str(self.denoise_steps),
                "seed": str(self.seed),
            }
        )

        res = reqPost(
            Constant.TRYON["local"],
            data=mutiPartBody,
            timeout=60,
            headers={"Content-Type": mutiPartBody.content_type},
        )

        if not res.ok:
            raise Exception(res.json())
        logger.info(f"{self.req.client.host}:{self.req.client.port} - tryOnLocal 성공적으로 예측")  # type: ignore
        return res._content  # type: ignore
