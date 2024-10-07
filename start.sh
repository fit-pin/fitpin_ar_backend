git pull

# 패키지 설치
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./.conda
conda env update -p .conda --prune

# 서버 실행
uvicorn Server:server --reload --port 8080 --host 0.0.0.0 --log-config ./src/log-config.yml --app-dir ./src
