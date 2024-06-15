# 이전 스크린 지우기
screen -X -S fitpin kill
git pull
# 스크린을 백그라운드로 시작
screen -dmS fitpin -L -Logfile log.txt bash -c 'uvicorn src.Server:server --reload --port 8080 --host 0.0.0.0'
