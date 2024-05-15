# 신체 측정 API
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
import uuid

router = APIRouter()


@router.post("/")
async def bodyMEAApi(anaFile: UploadFile):
    # 파일명 에서 확장자 구하기
    exte = anaFile.filename.split('.')[-1]

    # uuid 로 랜덤 파일명 부여
    fileName = f"{uuid.uuid4()}.{exte}"

    res = {
        "fileName": fileName,
        "result": {
            "top": 500,
            "leg": 500
        }
    }

    with open(f"src/res/{fileName}", "wb") as f:
        f.write(anaFile.file.read())
        
    
    return JSONResponse(content=res, media_type="application/json")
