# 프론트 결제 임시 백엔드용
# AR 백엔드 기능이 아니긴 함
import logging
from fastapi import APIRouter

router = APIRouter()
from fastapi.responses import RedirectResponse
from fastapi.requests import Request

logger = logging.getLogger("uvicorn.error")


@router.get("/{returnUrl}")
def payment(returnUrl: str, pg_token: str, req: Request):
    logger.info(f"{req.client.host}:{req.client.port} - 상태: {returnUrl}")  # type: ignore
    logger.info(f"{req.client.host}:{req.client.port} - 인증 토큰: {pg_token}")  # type: ignore

    # App scheme 으로 앱 실행 및 param 주기
    return RedirectResponse(f"fitpin://open?state={returnUrl}&pg_token={pg_token}")
