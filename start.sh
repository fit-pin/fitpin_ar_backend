git pull

# 패키지 설치
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./.conda
conda env update -p .conda --prune

# 서버 실행
# --root-path 리버스 프록시 uri와 동일하게
uvicorn Server:server --reload --port 80 --host 0.0.0.0 --log-config ./src/log-config.yml --app-dir ./src --root-path /ar-api
