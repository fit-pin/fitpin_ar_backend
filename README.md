# fitpin_backend_ar

핏핀 AR 백엔드

## API 사용법

**요청 메인 URL: `https://dmumars.kro.kr`**

> 예시 /bodymea/ = https://dmumars.kro.kr/bodymea/

-   모든 요청은 `multipart/form-data` 데이터로 요청합니다.
-   요청시 반드시 URL 끝에 `/` 붙어야 합니다

    > https://dmumars.kro.kr/bodymea (x) <br>
    > https://dmumars.kro.kr/bodymea/ (o)

**API 테스트 해보기**: https://dmumars.kro.kr/docs

## API 목록

### [**POST**] [/bodymea/](https://dmumars.kro.kr/bodymea/): 사진을 분석하여 체형을 측정 합니다

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

### [**POST**] [/try-on/](https://dmumars.kro.kr/try-on/): 채형 사진과 의류 이미지가 합성된 이미지를 리턴합니다.

> 해당 API 요청은 [IDM-VTON-FastAPI](https://github.com/fit-pin/IDM-VTON-FastAPI) 서버로 리다이렉트 되어 요청을 처리합니다

#### 요청

-   `Header`
    -   Content-Type: `multipart/form-data`
-   `Body`

    -   clothesImg: `File` - 의류사진 (바이너리)
    -   bodyFileName: `str` - AR 서버에 저장된 체형파일 명
    -   category: `str` - 의류 종류 (아래 값만 허용)

        ```text
        상의 | 하의 | 드레스
        ```

    -   is_checked: `bool = 기본값(True)` - Use auto-generated mask 설정
    -   is_checked_crop: `bool = 기본값(True)` - 크롭 사용
    -   denoise_steps: `int = 기본값(30)` - 노이즈 재거 단계
    -   seed: `int = 기본값(42)` - 랜덤시드

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

### [**POST**] [/clothesmea/](https://dmumars.kro.kr/clothesmea/): 오프라인 의류 측정을 진행합니다.

#### 요청

-   `Header`
    -   Content-Type: `multipart/form-data`
-   `Body`

    -   clothesImg: `File` - 누끼 따진 의류 이미지 (바이너리)
    -   clothesType: `String` - 의류 타입 (아래 값 만 허용)

        ```text
        반팔 |  긴팔 | 반팔 아우터 | 긴팔 아우터 | 조끼 | 슬링 | 반바지 | 긴바지 | 치마 | 반팔 원피스 | 긴팔 원피스 | 조끼 원피스 | 슬링 원피스
        ```

#### 정상응답 (code: 200)

-   Content-Type: `image/png`

> 측정된 의류 이미지 반환

#### 오류응답 (code: 500)

-   Content-Type: `application/json`

```js
{ "detail": "오류 메시지" }
```

-   오류 메시지
    -   `not_image`: 이미지가 아님
    -   `not_detection_card`: 사진에 카드를 감지하지 못함

### [**POST**] [/getnukki/](https://dmumars.kro.kr/getnukki/): 의류 이미지에 누끼를 땁니다.

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

## Development Guide

### ClothesMEA 의류 추가

> 현재는 긴팔, 긴바지, 반팔 밖에 못함

1. `CustumTypes.py` 하단에 해당 의류의 측정 부위 추가

    ```python
    변수명 = Literal["허리단면", "밑위", "엉덩이단면", "허벅지단면", "총장", "밑단단면"]
    """실측 타입 명시"""
    ```

2. `ClothesMEA.py` 하단 `LIST_KEY_POINTS`에 측정 부위별 키포인트를 명시

    > 측정 부위는 [해당링크](https://github.com/switchablenorms/DeepFashion2/blob/master/images/cls.jpg) 또는 테스트를 진행해서 조사

    - 키 값은 `CustumTypes.py` 에 명시해둔 `maskKeyPointsType` 안에 포함된 값이여야 함

        ```python
        "반바지": dict[1번과정에서 설정한 변수, tuple](
            {
                "부위": (0, 2),
            }
        ),
        ```

3. 추가적으로 `tuple` 관련 애러가 뜬다면 `COLOR_MAP` 변수가 부족한 거

    - 원하는 색상 검색해서 추가 하기 (B, G, R)

        ```python
        COLOR_MAP = (
            (181, 253, 120),
            (154, 153, 253),
            (221, 153, 0),
            (247, 247, 244),
            (250, 65, 137),
            (2, 78, 235),
            ...원하는거 추가
        )
        ```

## 빌드 및 테스트

### [모델파일 다운로드](https://huggingface.co/Seoksee/MY_MODEL_FILE/tree/main)

-   `.gitattributes` 를 제외한 모든 파일을 `src/model`에 넣기

### Docker 사용

```bash
docker run -it --name fitpin -p 80:80 ghcr.io/fit-pin/fitpin-ar-backend
```

## 개발환경

### 개발 언어 및 프레임워크

-   `Python 3.12.4`
-   `FastAPI`
-   `Pytorch`
-   `OpenCV`
