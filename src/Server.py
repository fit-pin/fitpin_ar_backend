from fastapi import FastAPI
from fastapi.responses import FileResponse
from os import path

from src.Constant import RES_DIR
from .api import BodyMEA, ClothesMEA, TryOn, GetNukki

server = FastAPI(redoc_url=None)

server.include_router(prefix="/bodymea", router=BodyMEA.router)
server.include_router(prefix="/clothesmea", router=ClothesMEA.router)
server.include_router(prefix="/try-on", router=TryOn.router)
server.include_router(prefix="/getnukki", router=GetNukki.router)


@server.get("/")
def root():
    return {"res": "ok"}

@server.get("/mills")
def testImg():
    return FileResponse(path.join(RES_DIR, "try.jpg"))