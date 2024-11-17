from typing import Literal


RES_DIR = "src/res/"
""" 리소스 폴더 경로"""

TRYON_MODE: Literal["huggingface", "local"] = "huggingface"
"""가상 피팅 모드"""

TRYON: dict[Literal["huggingface", "local"], str] = {
    "huggingface": "yisol/IDM-VTON",
    "local": "https://fitpin.kro.kr/ar-idm-api/api/idm/",
}
"""가상피팅 모드별 URL"""