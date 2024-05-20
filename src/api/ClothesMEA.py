# 의류 측정 API
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def clothesMEAApi():
    return {"res": "clothesMEA"}
