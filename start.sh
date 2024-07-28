git pull

# 패키지 설치
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./.conda
conda env update -p .conda --prune

# 스크린을 bash 명령으로 실행
uvicorn src.Server:server --reload --port 8080 --host 0.0.0.0
