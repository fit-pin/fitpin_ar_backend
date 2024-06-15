# 이전 스크린 지우기
screen -X -S fitpin kill
git pull
# 스크린을 bash 명령으로 실행
screen -S fitpin -L -Logfile log.txt bash -c 'uvicorn src.Server:server --reload --port 8080 --host 0.0.0.0'
