# fitpin_ar_backend

핏핀 AR 백엔드

## API 사용법

**요청 메인 URL: `http://dmumars.kro.kr:8080`**

> 예시 /bodymea/ = http://dmumars.kro.kr:8080/bodymea/

-   모든 요청은 `multipart/form-data` 데이터로 요청합니다.
-   요청시 반드시 URL 끝에 `/` 붙어야 합니다

    > http://localhost/bodymea (x) <br>
    > http://localhost/bodymea/ (o)

**API 테스트 해보기**: http://dmumars.kro.kr:8080/docs

## API 목록

### [**POST**] [/bodymea/](https://dmumars.kro.kr:8080/bodymea/): 사진을 분석하여 체형을 측정 합니다

#### 요청

-   `Header`
    -   Content-Type: `multipart/form-data`
-   `Body`

    -   anaFile: `File` - 체형을 분석할 파일 (바이너리)
    -   personKey: `Float` - 사용자 키(cm)

#### 정상응답 (code: 200)

-   Content-Type: `application/json`

```js
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

-   Content-Type: `application/json`

```js
{ "detail": "오류 메시지" }
```

-   오류 메시지
    -   `not_detection`: 사람 감지 안됨
    -   `many_detection`: 여러 사람 감지됨
    -   `keypoint_err`: 키포인트 검출실패
    -   `not_image`: 이미지가 아님

### [**POST**] [/try-on/](https://dmumars.kro.kr:8080/try-on/): 채형 사진과 의류 이미지가 합성된 이미지를 리턴합니다.

#### 요청

-   `Header`
    -   Content-Type: `multipart/form-data`
-   `Body`

    -   clothesImg: `File` - 누끼 따진 의류 이미지 (바이너리)
    -   clothesType: `String` - `"TOP"` | `"BOTTOM"`
    -   fileName: `Float` - 채형 측정시 나온 파일 이름
    -   personKey: `Float` - 사용자 키(cm)
    -   clothesLenth: `Float` - 옷, 바지 총장

#### 정상응답 (code: 200)

-   Content-Type: `image/png`

> 합성된 이미지 파일 리턴 (바이너리)

#### 오류응답 (code: 500)

-   Content-Type: `application/json`

```js
{ "detail": "오류 메시지" }
```

-   오류 메시지
    -   `not_exists_bodyImg`: 채형 측정 이미지가 없음
    -   `not_detection`: 사람 감지 안됨
    -   `many_detection`: 여러 사람 감지됨

### [**POST**] [/getnukki/](https://dmumars.kro.kr:8080/getnukki/): 의류 이미지에 누끼를 땁니다.

> 관리자 기능

#### 요청

-   `Header`
    -   Content-Type: `multipart/form-data`
-   `Body`

    -   clothesImg: `File` - 누끼 따려는 이미지

#### 정상응답 (code: 200)

-   Content-Type: `image/png`

> 합성된 이미지 파일 리턴 (바이너리)

#### 오류응답 (code: 500)

-   Content-Type: `application/json`

```js
{ "detail": "오류 메시지" }
```

-   오류 메시지

    > 정의된 오류는 없음

## 빌드 및 테스트

### Docker 사용

1. Dockerfile 이미지 빌드

    ```bash
    docker buildx build --load -t fitpin .
    ```

2. 컨테이너 생성 & 실행

    ```bash
    docker run -it --name fitpin -p 8080:8080 fitpin
    ```

## 개발환경

### 개발 언어 및 프레임워크

-   `Python 3.12.4`
-   `FastAPI`
-   `OpenCV`
-   `Docker`
