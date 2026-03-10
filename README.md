# SAR Bounding Box Test Guide

`test.py`는 MMDetection `DetInferencer`를 사용해 이미지/폴더 입력에 대해 바운딩박스를 추론하고 시각화 결과를 저장하는 최소 실행 스크립트입니다.

## 1) 사전 준비

실행 전에 **가상환경 세팅이 완료**되어 있어야 합니다.

- Python 가상환경(venv/conda) 활성화
- `mmdet` 설치
- `mmcv` 컴파일 확장 포함 버전 설치 필요 (`mmcv._ext` 누락 시 실행 불가)
- `msfa` import 가능한 경로 확인 (기본값은 `SARDet_100K-main/SARDet_100K-main/MSFA`)

## 2) 중요한 주의사항

### Config와 모델(.pth) 일치
- `--config`와 `--weights`는 **같은 학습 실험에서 나온 짝**이어야 합니다.
- 구조가 다른 config + checkpoint를 섞으면 로딩 에러 또는 잘못된 추론 결과가 나올 수 있습니다.
- 클래스 수/헤드 구조가 다르면 특히 문제가 자주 발생합니다.

### 입력 경로(`--input`) 주의
- 파일 또는 폴더 경로를 넣을 수 있습니다.
- 경로가 실제로 존재하지 않으면 스크립트가 바로 종료됩니다.
- 한글/공백 경로는 따옴표로 감싸는 것을 권장합니다.

### 출력 경로(`--out-dir`) 주의
- 기본 출력 폴더는 `outputs`입니다.
- `--no-save-vis --no-save-pred`를 동시에 쓰면 저장이 비활성화되어 `out_dir`는 무시됩니다.
- 기존 `out-dir` 아래 결과 파일은 실행 조건에 따라 덮어써질 수 있으니 폴더를 분리해 관리하는 것을 권장합니다.

## 3) 실행 방법

### 기본값으로 실행
```bash
python test.py
```

### 수동 지정 실행
```bash
python test.py --input <이미지/폴더> --config config.py --weights best_coco_bbox_mAP_epoch_12.pth --out-dir outputs
```

## 4) 자주 쓰는 옵션

- `--device cuda:0` 또는 `--device cpu`
- `--pred-score-thr 0.3` (bbox score threshold)
- `--show` (팝업으로 시각화 표시)
- `--no-save-vis` (시각화 이미지 저장 안 함)
- `--no-save-pred` (예측 JSON 저장 안 함)

## 5) 기본 경로 동작

`python test.py`로 실행하면, 코드 내부 기본값을 사용합니다.

- 입력: `tx_1.2x_to_256x256_8bit`
- 설정: `config.py`
- 가중치: `best_coco_bbox_mAP_epoch_12.pth`
- 출력: `outputs`
