# 의류 합성 이미지 생성 API
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def vrTryonApi():
    return {
        "res": "vrTryon"
    }