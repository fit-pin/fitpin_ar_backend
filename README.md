# fitpin_ar_backend

핏핀 AR 백엔드

## API 목록

### `/bodymea`: 사진을 분석하여 체형을 측정 합니다

#### 요청

-   `Header`
    -   Method: `Post`
    -   Content-Type: `multipart/form-data`
-   `Body`

    -   anaFile: `file` - 체형을 분석할 파일 (바이너리)
    -   personKey: `float` - 사용자 키(cm)

#### 정상응답 (code: 200)

```json
{
    "fileName": "2c49f715-67b8-40ec-86a2-b9d3e2875923.jpg", //저장된 파일 명
    "result": {
        "armSize": 58.37, // 팔 길이
        "shoulderSize": 32.64, // 어께 너비
        "bodySize": 52.63, // 몸길이
        "legSize": 63.82 // 다리 길이
    }
}
```

#### 오류응답 (code: 500)

```json
{ "detail": "오류 메시지" }
```

-   오류 메시지
    -   `not_detection`: 사람 감지 안됨
    -   `many_detection`: 여러 사람 감지됨
    -   `keypoint_err`: 키포인트 검출실패
    -   `not_image`: 이미지가 아님

## 빌드 및 테스트

1. poetry 설치
    ```
    pip install poetry
    ```
2. `poetry install`: 패키지 설치
3. `poetry run fastapi [dev, run] src/Server.py`: 서버 실행
    - 개발시: `dev`
    - 배포시: `run`
