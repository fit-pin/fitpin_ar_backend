from fastapi import FastAPI
from .api import BodyMEA, ClothesMEA, VRTryon

server = FastAPI()

server.include_router(prefix="/bodymea", router=BodyMEA.router)
server.include_router(prefix="/clothesmea", router=ClothesMEA.router)
server.include_router(prefix="/vrtryon", router=VRTryon.router)

@server.get("/")
def root():
    return {
        "res": "ok"
    }
