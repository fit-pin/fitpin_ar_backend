from fastapi import FastAPI
import json

server = FastAPI()

@server.get("/")
def root():
    return {
        "res": "ok"
    }
