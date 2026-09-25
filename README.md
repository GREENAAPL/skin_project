# EfficientNet-B0 피부 병변 7-class 분류

ISIC 2018 Task 3/HAM10000 피부 병변 사진을 `MEL`, `NV`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC` 중 하나로 분류하는 **단일 EfficientNet-B0 이미지 모델**입니다. 정상 밝기 점수와 촬영 밝기가 달라졌을 때의 성능 변화를 함께 측정합니다.

이 저장소의 모델은 **교육·연구용이며 의료 진단에 사용할 수 없습니다.**

## 최종 결과

모든 실험은 seed 42, 동일한 lesion-disjoint dev split 2,026장, 20 epochs에서 진행했습니다. 주 지표는 7개 클래스를 같은 비중으로 평가하는 Macro-F1입니다.

| 실험 | 본체 LR | 분류기 LR | 정상 Macro-F1 | 변형 밝기 평균 | 최악 밝기 F1 |
|---|---:|---:|---:|---:|---:|
| **LR-0** | **0.00020** | **0.00020** | **0.7641** | **0.6501** | 0.5113 |
| LR-1 | 0.00005 | 0.00020 | 0.6846 | 0.6000 | 0.4907 |
| LR-2 | 0.00005 | 0.00050 | 0.6795 | 0.5893 | 0.4780 |
| LR-3 | 0.00001 | 0.00050 | 0.5613 | 0.4944 | 0.3957 |
| LR-4 | 0.00010 | 0.00050 | 0.7185 | 0.6252 | **0.5277** |

최종 모델은 **EfficientNet 본체와 분류기에 모두 learning rate 0.0002를 사용한 LR-0**입니다. 새 분류기의 LR을 크게 하고 사전학습 본체의 LR을 작게 만드는 방법도 실험했지만, 이번 데이터와 20-epoch 조건에서는 기존 동일 LR이 가장 높은 Macro-F1을 기록했습니다.

![EfficientNet learning-rate 비교](docs/figures/efficientnet_lr_comparison.png)

자세한 모델 설명과 결과 해석은 [EfficientNet.md](EfficientNet.md), 전체 연구 방법은 [method.md](method.md), 정확한 결과 CSV는 [docs/results/efficientnet_lr_comparison.csv](docs/results/efficientnet_lr_comparison.csv)에 있습니다.

## 최종 모델 설정

- backbone: ImageNet 사전학습 EfficientNet-B0
- 입력: RGB 320×320
- 출력: 7 classes
- fine-tuning: 마지막 4개 feature block + classifier
- 학습 가능한 parameter: 3,707,855개
- metadata: 사용하지 않음
- optimizer: AdamW
- backbone/classifier learning rate: 각각 0.0002
- weight decay: 0.0001
- scheduler: cosine annealing
- loss: class-weighted cross entropy + label smoothing 0.05
- batch size: 32
- epochs: 20
- best epoch: 17
- hardware: NVIDIA GeForce RTX 4070 12 GB

## 데이터

[ISIC 2018 Challenge Task 3](https://challenge.isic-archive.com/data/) 학습 이미지 10,015장을 사용합니다. 데이터 라이선스는 CC BY-NC 4.0이며 저장소에 원본 데이터를 포함하지 않습니다.

| split | 이미지 | 서로 다른 병변 | MEL | NV | BCC | AKIEC | BKL | DF | VASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7,989 | 5,976 | 888 | 5,339 | 409 | 266 | 878 | 94 | 115 |
| dev | 2,026 | 1,494 | 225 | 1,366 | 105 | 61 | 221 | 21 | 27 |

같은 병변을 여러 번 촬영한 사진이 있으므로 `lesion_id`를 기준으로 train/dev를 나눴습니다. 두 split의 병변 중복은 0개입니다. `lesion_id`는 누수 방지에만 사용하며 모델 입력에는 넣지 않습니다.

## 설치

Windows PowerShell과 Python 3.10 이상을 기준으로 작성했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 데이터 다운로드와 준비

다운로드 스크립트는 약 2.8 GB의 이미지 파일을 병렬 다운로드하고 크기·해시·이미지 수를 확인합니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_isic2018.ps1 -Connections 8
powershell -ExecutionPolicy Bypass -File scripts\prepare_isic2018.ps1
```

## 최종 모델 학습

```powershell
python -m skin_project.training `
  --config configs/isic2018_7class_efficientnet_b0_320.yaml
```

최고 checkpoint는 다음 경로에 저장됩니다.

```text
outputs/checkpoints/isic2018_7class_efficientnet_b0_320.pt
```

## 5개 Learning Rate 실험 재현

아래 스크립트는 LR-0부터 LR-4까지 순서대로 학습하고, 각 checkpoint에 7개 밝기 조건을 적용한 뒤 비교 CSV와 그래프를 생성합니다. 이미 완료된 실험은 자동으로 재사용합니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_efficientnet_lr_experiments.ps1
```

결과는 다음 위치에 생성됩니다.

```text
outputs/experiments/efficientnet_lr/efficientnet_lr_comparison.csv
outputs/experiments/efficientnet_lr/efficientnet_lr_comparison.png
```

## 밝기 강건성 평가

동일한 dev 이미지에 밝기 계수 `0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50`을 적용합니다. `1.00`은 원본, 작은 값은 어두운 영상, 큰 값은 밝은 영상입니다.

```powershell
python -m skin_project.robustness `
  --config configs/isic2018_7class_efficientnet_b0_320.yaml `
  --checkpoint outputs/checkpoints/isic2018_7class_efficientnet_b0_320.pt `
  --output-dir outputs/robustness/isic2018_7class_efficientnet_b0_320/dev
```

최종 모델의 밝기별 Macro-F1은 다음과 같습니다.

| 밝기 계수 | 0.50 | 0.70 | 0.85 | 1.00 | 1.15 | 1.30 | 1.50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Macro-F1 | 0.5113 | 0.6678 | 0.7476 | **0.7641** | 0.7182 | 0.6937 | 0.5617 |

## 테스트

```powershell
pytest -q
```

테스트는 데이터 누수 방지, Dataset/DataLoader, 밝기 변환, metric, checkpoint, EfficientNet 출력과 분리 learning-rate parameter group을 검증합니다.

## 주요 파일

```text
EfficientNet.md                              모델 구조와 LR 실험 해석
method.md                                    전체 연구 방법
configs/isic2018_7class_efficientnet_b0_320.yaml 최종 모델 설정
configs/isic2018_efficientnet_lr*.yaml       비교 실험 설정
scripts/run_efficientnet_lr_experiments.ps1  전체 실험 실행
scripts/compare_efficientnet_lr_results.py   결과 표·그래프 생성
src/skin_project/isic2018.py                 lesion-disjoint split
src/skin_project/training.py                 학습과 분리 LR optimizer
src/skin_project/robustness.py               밝기 변화 평가
```

`data/`, `outputs/`, `runs/`, `*.pt`는 용량과 데이터 라이선스 때문에 Git에 올리지 않습니다. 다운로드·학습 스크립트로 재생성할 수 있습니다.

## 참고 문헌

- ISIC Challenge, [ISIC 2018 data](https://challenge.isic-archive.com/data/)
- Tschandl et al., *The HAM10000 dataset*, [Scientific Data 5, 180161 (2018)](https://doi.org/10.1038/sdata.2018.161)
- Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
