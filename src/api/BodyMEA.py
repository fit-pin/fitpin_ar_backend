# 신체 측정 API
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def bodyMEAApi():
    return {
        "res": "bodyMEA"
    }