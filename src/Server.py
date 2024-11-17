from fastapi import FastAPI, Request

from api import BodyMEA, ClothesMEA, TryOn, GetNukki, Payment
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)

server = FastAPI(redoc_url=None, docs_url=None)

server.include_router(prefix="/bodymea", router=BodyMEA.router)
server.include_router(prefix="/clothesmea", router=ClothesMEA.router)
server.include_router(prefix="/try-on", router=TryOn.router)
server.include_router(prefix="/getnukki", router=GetNukki.router)
server.include_router(prefix="/payment", router=Payment.router)


@server.get("/")
def root():
    return {"res": "ok"}


# 리버스 프록시를 위해 자체 docs 구현
@server.get("/docs", include_in_schema=False)
async def swaggerDocs(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + server.openapi_url  # type: ignore
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="FastAPI - Swagger UI",
    )
