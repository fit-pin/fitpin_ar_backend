# 의류 측정 API
import logging
import math
import threading
from typing import Any, Literal
from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile
import numpy as np
import cv2 as cv

from torch import (
    Tensor,
    cat,
    load,
    device as Device,
    stack,
    tensor,
    zeros,
    abs as torch_abs,
)
from torch.cuda import is_available
from torch.nn import DataParallel

from ultralytics import YOLO
from src.lib import pose_hrnet

from src import Utills
from src import CustumTypes


router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.post("/")
def clothesMEAApi(
    clothesImg: UploadFile,
    req: Request,
    clothesType: CustumTypes.maskKeyPointsType = Form(),
):
    # 이미지인지 예외처리
    try:
        clothesImg_encoded = np.fromfile(clothesImg.file, dtype=np.uint8)
        clothesImg_decode = cv.imdecode(clothesImg_encoded, cv.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=500, detail="not_image")

    work = WorkClothesMEA(clothesImg_decode, clothesType, req)
    try:
        resultImg = work.getClothesMEA()
    except Exception as e:
        logger.error(f"{req.client.host}:{req.client.port} - 애러: {e}")  # type: ignore
        raise HTTPException(status_code=500, detail=f"{e}")

    return Response(
        cv.imencode(".png", resultImg)[1].tobytes(), media_type="image/jpeg"
    )


class WorkClothesMEA:
    def __init__(
        self,
        clothesImg: cv.typing.MatLike,
        clothesType: CustumTypes.maskKeyPointsType,
        req: Request,
    ):
        device = Device("cpu")
        if is_available():
            device = Device("cuda:0")

        self.clothesImg = clothesImg
        """의류 이미지"""

        self.clothesType: CustumTypes.maskKeyPointsType = clothesType
        """의류 타입"""

        self.req = req
        """요청"""

        self.device = device
        """torch 디바이스"""

    def getClothesMEA(self, is_multiThread=True) -> cv.typing.MatLike:
        """
        의류의 실측 결과 이미지를 반환 합니다

        Args:
            is_multiThread (bool, optional): 멀티쓰레딩 사용 여부

        Returns:
            cv.typing.MatLike: 완성된 이미지
            
        Raises:
            Exception:
                - not_detection_card: 카드 감지 불가
        """
        # 멀티쓰레딩 결과 저장용
        syncData: dict[Literal["getCardHeight"] | Literal["getKeyPoints"], Tensor] = (
            dict()
        )

        # 예외 기록
        err_flags = []
        if is_multiThread:
            # __getCardHeight 쓰레드 생성
            CardHeightThread = threading.Thread(
                target=self.__getCardHeight, args=(syncData, err_flags)
            )
            # __getClothesPoints 쓰레드 생성
            keyPointThread = threading.Thread(
                target=self.__getClothesPoints, args=(syncData, err_flags)
            )

            # 쓰레드 시작
            CardHeightThread.start()
            keyPointThread.start()

            # 완료대기
            CardHeightThread.join()
            keyPointThread.join()
        else:
            self.__getCardHeight(syncData, err_flags)
            self.__getClothesPoints(syncData, err_flags)

        # 예외 발생 시
        if err_flags:
            raise Exception(err_flags[0])

        # 의류 점
        clothes_point = syncData["getKeyPoints"]
        # 픽셀상 세로 크기
        card_px = float(syncData["getCardHeight"])

        # 긴팔: TopMeaType
        # 긴바지: BottomMeaType
        MEAData = self.__getMEApoints(clothes_point, self.clothesType)

        # 카드 크기랑, 의류 키포인트 가지고 실측 길이 구하기
        realDist_Dict: dict[Any, float] = {}
        for idx in MEAData.keys():
            pixelSize = Utills.distance(MEAData[idx])
            realDist_Dict[idx] = Utills.findRealSize(
                WorkClothesMEA.CARD_SIZE[1], card_px, pixelSize
            )

        logger.info(
            f"{self.req.client.host}:{self.req.client.port} - 의류 추정 완료: {realDist_Dict}"  # type: ignore
        )

        """이쪽 부터는 시각화"""
        centerPose = Tensor()
        for i, part in enumerate(MEAData.keys()):
            points = MEAData[part]

            # 각 파트별 중점 좌표를 2차원 텐서로 저장하는 코드
            center = MEAData[part].mean(dim=0)
            centerPose = cat((centerPose, center.unsqueeze(0)))

            # 점과 라인 찍기
            for k, point in enumerate(points):
                cv.circle(
                    self.clothesImg,
                    (int(point[0]), int(point[1])),
                    2,
                    WorkClothesMEA.COLOR_MAP[i],
                    10,
                )
                if k < len(points) - 1:
                    pt1 = (int(points[k][0]), int(points[k][1]))
                    pt2 = (int(points[k + 1][0]), int(points[k + 1][1]))
                    cv.line(self.clothesImg, pt1, pt2, WorkClothesMEA.COLOR_MAP[i], 5)

        # 각각의 중점 좌표가 똑같은 경우 텍스트 겹침 현상을 해결하고자 만든 반복문
        # 자신의 좌표가 centerPose에 저장된 값과 +- 100 이하면 +100 해주는 코드
        for i, part in enumerate(MEAData.keys()):
            strSize = f"{round(realDist_Dict[part], 2)}cm"

            # i값의 중점 x 좌표를 모든 좌표와 뺄샘 연산을 진행
            x_per = torch_abs(centerPose - centerPose[i][0])
            # i값의 중점 y 좌표를 모든 좌표와 뺄샘 연산을 진행
            y_per = torch_abs(centerPose - centerPose[i][1])

            # 모든 x, y 에 대해 100 이하인지를 저장하는 bool 마스크를 생성
            mask_x = x_per < 100
            mask_y = y_per < 100

            # mask_x와 mask_y 에 대한 or 연산 진행 (x_per < 100 or y_per < 100) 이게 tenor 에서 안됨
            mask = mask_x.logical_or(mask_y)

            # 자기 자신을 True 로 처리하지 못하게 처리
            mask[i] = False

            # 모든 x 좌표 또는 y 좌표에 True 가 있는지 판단
            is_Overlap = mask.any(dim=0)

            # 이게 or 로 두면 같은 x 좌표에 다른 y 값을 가진경우라도 통과할 수 있다
            if is_Overlap[0] and is_Overlap[1]:
                centerPose[i] += 100

            cul_points = centerPose[i]

            cv.putText(
                self.clothesImg,
                strSize,
                (int(cul_points[0] + 30), int(cul_points[1]) - 30),
                cv.FONT_HERSHEY_PLAIN,
                5,
                WorkClothesMEA.COLOR_MAP[i],
                5,
            )

        return self.clothesImg

    def __getCardHeight(self, syncData: dict, err_flags: list):
        """
        카드의 세로 px 크기를 반환합니다.</br>
        결과 반환은 모두 `syncData` 안에 포함 됩니다.</br>
        예외가 발생하면 `err_flags` 에 기록됩니다

        Returns:
            Tensor: syncData["getCardHeight"]

        Raises:
            str: err_flags
                - not_detection_card: 카드 감지 불가
        """
        result = WorkClothesMEA.CARDPOINT_MODEL.predict(self.clothesImg)[0]

        if not len(result.obb.cls):  # type: ignore
            err_flags.append("not_detection_card")
            return
        
        # 예측확율 가장 좋은거 선택
        max_value = 0.0
        max_index = 0
        for i, card in enumerate(result.obb.conf):  # type: ignore
            if max_value < float(card):
                max_value = float(card)
                max_index = i

        points: Tensor = result.obb.xywhr[max_index]  # type: ignore
        w, h = points[2:4]

        # 제일 작은 값이 세로 이므로
        height = h
        if int(w) < int(h):
            height = w

        syncData["getCardHeight"] = height

    def __getClothesPoints(self, syncData: dict, err_flags: list):
        """
        의류의 키포인트를 반환합니다.</br>
        결과 반환은 모두 `syncData` 안에 포함 됩니다.</br>
        예외가 발생하면 `err_flags` 에 기록됩니다

        Returns:
            Tensor: syncData["getKeyPoints"]
        """
        # 모델 불러오기
        model = pose_hrnet.get_pose_net()
        model.load_state_dict(
            load(WorkClothesMEA.KEYPOINT_MODEL_CKP, map_location=self.device),
            strict=True,
        )

        # to() = cpu() 쓸지 cuda() 쓸지 device 메게 변수로 알아서 처리
        model = DataParallel(model).to(self.device)
        model.eval()

        # 이미지 크기를 288x384 로 변경
        reSizeImage, padding = Utills.resizeWithPad(self.clothesImg, (288, 384))

        print(f"getKeyPoints: 이미지에 적용된 패딩: {padding}")

        # 이미지 정규화 하기
        normaImg = Utills.getNormalizimage(reSizeImage)

        # 해당 __call__  메소드 구현은 부모에 구현되있는데 그쪽에서 forward 함수를 호출하도록 설계함
        # 따라서 pose_hrnet.py 에 forward() 함수를 찾아가면 됨
        res = model(normaImg)

        # 키포인트 추려내기
        keyPoints = self.__getKeyPointsResult(res, clothType=self.clothesType)

        pointPadding = 2
        # 히트맵 사이즈 보정
        scaling_factor_x = WorkClothesMEA.IMG_SIZE[1] / WorkClothesMEA.HEATMAP_SIZE[0]
        scaling_factor_y = WorkClothesMEA.IMG_SIZE[0] / WorkClothesMEA.HEATMAP_SIZE[1]

        result_points = Tensor()
        for points in keyPoints[0]:
            # 288x384 에서 표시되야 하는 점
            joint_x = pointPadding + points[0] * scaling_factor_x
            joint_y = pointPadding + points[1] * scaling_factor_y

            # 원본 이미지 비율에서 보정 되야 하는 크기
            ratio_x = self.clothesImg.shape[1] / (
                WorkClothesMEA.IMG_SIZE[0] - padding["left"] - padding["right"]
            )
            ratio_y = self.clothesImg.shape[0] / (
                WorkClothesMEA.IMG_SIZE[1] - padding["top"] - padding["bottom"]
            )
            if points[0] or points[1]:
                # 최종적으로 패딩 값에 따른 점 위치 수정
                final_x = joint_x * ratio_x - (ratio_x * padding["left"])
                final_y = joint_y * ratio_y - (ratio_y * padding["top"])

                # 2차원으로 변경
                temp = tensor([final_x, final_y]).unsqueeze(0)
                result_points = cat((result_points, temp))

        syncData["getKeyPoints"] = result_points

    def __getKeyPointsResult(
        self,
        predOutput: Tensor,
        is_mask=True,
        flipTest=False,
        clothType: CustumTypes.maskKeyPointsType = "반팔",
    ):
        """
        정규화된 이미지로 키포인트를 얻습니다

        Args:
            predOutput (Tensor): 정규화 이미지 텐서
            is_mask (bool, optional): 의류 타입별 채널 마스크 여부 (안하면 오차 keyPoint가 발생)
            flipTest (bool, optional): flipTest 여부
            clothType (int, optional): 의류타입

        Returns:
            Tensor: 의류 키포인트
        """

        if is_mask:
            channel_mask = zeros((1, 294, 1)).to(self.device).float()

            rg = WorkClothesMEA.MASK_KEY_POINTS[clothType]
            index = (
                tensor(
                    [list(range(rg[0], rg[1]))],
                    device=channel_mask.device,
                    dtype=channel_mask.dtype,
                )
                .transpose(1, 0)
                .long()
            )
            channel_mask[0].scatter_(0, index, 1)

            predOutput = predOutput * channel_mask.unsqueeze(3)

        heatmap_height = WorkClothesMEA.HEATMAP_SIZE[0]
        heatmap_width = WorkClothesMEA.HEATMAP_SIZE[1]

        batch_heatmaps = predOutput.detach().cpu().numpy()

        assert isinstance(
            batch_heatmaps, np.ndarray
        ), "batch_heatmaps should be numpy.ndarray"
        assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds_local = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds_local[:, :, 0] = (preds_local[:, :, 0]) % width
        preds_local[:, :, 1] = np.floor((preds_local[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds_local *= pred_mask

        for n in range(preds_local.shape[0]):
            for p in range(preds_local.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(preds_local[n][p][0] + 0.5))
                py = int(math.floor(preds_local[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    preds_local[n][p] += np.sign(diff) * 0.25

        return preds_local

    def __getMEApoints(
        self, resultPoint: Tensor, type: CustumTypes.maskKeyPointsType
    ) -> dict[Any, Tensor]:
        """
        실측 크기에 유의미한 좌표값을 얻습니다.</br>

        긴팔 Type: `dict[TopKeyPointsType, Tensor]`</br>
        긴바지 Type: `dict[BottomKeyPointsType, Tensor]`</br>
        Args:
            resultPoint (Tensor): 전체 결과 키포인트
            type (Literal["긴팔", "긴바지"]):

        Returns:
            dict[Any, Tensor]: 결과반환
        """
        resultDict = {}
        if type == "긴팔":
            for key in WorkClothesMEA.TOP_KEY_POINTS.keys():
                # 이게 리스트 안에 Tensor 가 있어서 stack() 함수로 전체를 변환해야함
                resultDict[key] = stack(
                    [resultPoint[index] for index in WorkClothesMEA.TOP_KEY_POINTS[key]]
                )
        elif type == "긴바지":
            for key in WorkClothesMEA.BOTTOM_KEY_POINTS.keys():
                resultDict[key] = stack(
                    [
                        resultPoint[index]
                        for index in WorkClothesMEA.BOTTOM_KEY_POINTS[key]
                    ]
                )

        return resultDict

    # 상수정의
    CARDPOINT_MODEL = YOLO("src/model/Clothes-Card.pt")
    """카드 감지 모델"""

    KEYPOINT_MODEL_CKP = "src/model/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth"
    """의류 키포인트 모델 경로"""

    IMG_SIZE = (288, 384)
    """의류 입/출력 이미지 사이즈"""

    HEATMAP_SIZE = (96, 72)
    """히트맵 사이즈"""

    CARD_SIZE = (8.56, 5.398)
    """신용카드 규격 (가로, 세로)"""

    COLOR_MAP = ((181, 253, 120), (154, 153, 253), (221, 153, 0), (247, 247, 244))
    """키포인트 컬러맵"""

    MASK_KEY_POINTS: dict[CustumTypes.maskKeyPointsType, tuple] = {
        "반팔": (0, 25),
        "긴팔": (25, 58),
        "반팔 아우터": (58, 89),
        "긴팔 아우터": (89, 128),
        "조끼": (128, 143),
        "슬링": (143, 158),
        "반바지": (158, 168),
        "긴바지": (168, 182),
        "치마": (182, 190),
        "반팔 원피스": (190, 219),
        "긴팔 원피스": (219, 256),
        "조끼 원피스": (256, 275),
        "슬링 원피스": (275, 294),
    }
    """
    키포인트 마스크
    [예시 참고 사진](https://github.com/switchablenorms/DeepFashion2/blob/master/images/cls.jpg)
    """

    TOP_KEY_POINTS: dict[CustumTypes.TopKeyPointsType, tuple] = {
        "어깨너비": (6, 32),
        "가슴단면": (15, 23),
        "소매길이": (32, 31, 30, 29, 28),
        "총장": (0, 19),
    }
    """긴팔 실측 키포인트"""

    BOTTOM_KEY_POINTS: dict[CustumTypes.BottomKeyPointsType, tuple] = {
        "허리단면": (0, 2),
        "밑위": (1, 8),
        "엉덩이단면": (3, 13),
        "허벅지단면": (8, 13),
        "총장": (0, 3, 4, 5),
        "밑단단면": (5, 6),
    }
    """긴바지 실측 키포인트"""
