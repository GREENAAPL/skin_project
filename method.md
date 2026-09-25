# Method: EfficientNet-B0 피부 병변 분류와 Learning Rate 실험

## 1. 연구 질문

1. 피부 병변 이미지만으로 ISIC 2018의 7개 클래스를 얼마나 잘 분류할 수 있는가?
2. ImageNet 사전학습 EfficientNet 본체와 새 분류기에 서로 다른 learning rate를 주면 성능이 향상되는가?
3. 촬영 밝기가 바뀌었을 때 최종 모델의 성능은 얼마나 떨어지는가?

최종 시스템은 앙상블이나 metadata를 사용하지 않는 **단일 EfficientNet-B0 이미지 모델**이다.

## 2. 데이터

ISIC 2018 Task 3/HAM10000 RGB 피부경 이미지 10,015장과 7-class 정답을 사용한다.

| index | code | 의미 |
|---:|---|---|
| 0 | MEL | melanoma |
| 1 | NV | melanocytic nevus |
| 2 | BCC | basal cell carcinoma |
| 3 | AKIEC | actinic keratosis / Bowen disease |
| 4 | BKL | benign keratosis |
| 5 | DF | dermatofibroma |
| 6 | VASC | vascular lesion |

### Lesion-disjoint split

한 병변에 여러 이미지가 있을 수 있으므로 이미지 단위 무작위 분할은 비슷한 사진이 train과 dev에 동시에 들어가는 누수를 만들 수 있다. 공식 `LesionGroupings.csv`를 사용해 병변 단위로 계층화한 80:20 split을 만들었다.

- train: 7,989 images / 5,976 lesions
- dev: 2,026 images / 1,494 lesions
- train/dev `lesion_id` overlap: 0
- seed: 42

`lesion_id`와 진단 확정 방법은 분할과 추적에만 사용한다. 나이, 성별, 위치를 포함한 metadata는 모델 입력에 사용하지 않는다.

## 3. 전처리와 증강

이미지를 RGB로 변환하고 320×320으로 처리한 뒤 ImageNet 평균과 표준편차로 정규화한다.

- random resized crop: scale 0.85–1.00
- horizontal flip
- vertical flip
- rotation: 최대 20도
- 학습 brightness jitter: 사용하지 않음

LR 비교에서 augmentation을 포함한 모든 조건을 고정했다.

## 4. 모델

ImageNet 사전학습 EfficientNet-B0를 사용한다. 원래 classifier를 dropout 0.3과 7-output linear layer로 교체했다. 앞쪽 encoder는 고정하고 마지막 4개 feature block과 classifier를 fine-tuning한다.

```text
320×320 RGB image
→ EfficientNet-B0 feature blocks
→ global average pooling
→ dropout 0.3
→ linear classifier
→ 7 logits
```

학습 가능한 parameter는 3,707,855개다.

## 5. Loss와 최적화

NV가 많고 DF/VASC가 적은 불균형 데이터이므로 train class frequency의 역수에 비례하는 class weight를 사용한다.

```text
w_c = (1 / n_c) / sum_k(1 / n_k) × C
```

Loss는 weighted cross entropy와 label smoothing 0.05의 조합이다.

| 항목 | 값 |
|---|---|
| optimizer | AdamW |
| weight decay | 0.0001 |
| scheduler | cosine annealing |
| epochs | 20 |
| batch size | 32 |
| checkpoint 선택 | dev Macro-F1 최고 epoch |
| hardware | NVIDIA GeForce RTX 4070 12 GB |

## 6. Differential Learning Rate 구현

학습 가능한 encoder parameter와 classifier parameter를 서로 다른 optimizer group으로 분리했다.

```python
optimizer = torch.optim.AdamW([
    {"params": backbone_parameters, "lr": backbone_learning_rate},
    {"params": classifier_parameters, "lr": classifier_learning_rate},
])
```

YAML에서 두 값을 독립적으로 설정한다. 별도 값이 없으면 기존 `learning_rate`를 사용해 이전 설정과 호환한다.

## 7. LR 실험 설계

두 LR을 제외한 데이터, seed, model, augmentation, batch size, epochs, loss와 scheduler를 고정했다.

| 실험 | 본체 LR | 분류기 LR | 비교 목적 |
|---|---:|---:|---|
| LR-0 | 0.00020 | 0.00020 | 기존 기준점 |
| LR-1 | 0.00005 | 0.00020 | 본체 LR만 감소 |
| LR-2 | 0.00005 | 0.00050 | 분류기 LR 증가 |
| LR-3 | 0.00001 | 0.00050 | 매우 작은 본체 LR |
| LR-4 | 0.00010 | 0.00050 | 중간 본체 LR |

LR-0은 앞서 동일한 조건으로 완료한 checkpoint를 재사용했고 LR-1~LR-4는 새로 학습했다. LR-0의 두 group은 같은 LR이므로 기존 단일 optimizer group과 parameter별 AdamW update가 동일하다.

## 8. 평가지표

주 지표는 Macro-F1이다.

```text
F1_c = 2 × precision_c × recall_c / (precision_c + recall_c)
Macro-F1 = (1 / C) × sum_c F1_c
```

Macro-F1은 데이터가 많은 NV뿐 아니라 적은 DF와 VASC도 같은 비중으로 평가한다. balanced accuracy와 train-dev gap은 보조 지표로 사용한다.

## 9. LR 실험 결과

| 실험 | 최고 epoch | Train F1 | Dev Macro-F1 | Train-dev gap | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| **LR-0** | 17 | 0.9165 | **0.7641** | 0.1524 | 0.7958 |
| LR-1 | 16 | 0.7782 | 0.6846 | 0.0936 | 0.8041 |
| LR-2 | 16 | 0.7806 | 0.6795 | 0.1011 | **0.8122** |
| LR-3 | 17 | 0.5926 | 0.5613 | 0.0313 | 0.7714 |
| LR-4 | 17 | 0.8513 | 0.7185 | 0.1328 | 0.8078 |

LR-0이 주 지표에서 가장 높았다. LR-3은 train 점수도 낮아 본체 LR이 지나치게 작을 때의 underfitting을 보여준다. LR-2의 balanced accuracy는 높았지만 precision까지 반영하는 Macro-F1이 낮았으므로 최종 모델로 선택하지 않았다.

## 10. 밝기 강건성 평가

동일한 dev 이미지에 고정 brightness factor를 적용한다.

```text
I_alpha = clip(alpha × I, 0, 1)
alpha ∈ {0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50}
```

`alpha=1.00`은 원본이다. 1보다 작으면 어둡고 크면 밝다. 모든 조건에서 이미지 구성과 정답은 같다.

| 실험 | 정상 F1 | 변형 평균 F1 | 최악 factor | 최악 F1 | 최대 상대 하락률 |
|---|---:|---:|---:|---:|---:|
| **LR-0** | **0.7641** | **0.6501** | 0.50 | 0.5113 | 33.08% |
| LR-1 | 0.6846 | 0.6000 | 1.50 | 0.4907 | 28.31% |
| LR-2 | 0.6795 | 0.5893 | 1.50 | 0.4780 | 29.65% |
| LR-3 | 0.5613 | 0.4944 | 1.50 | 0.3957 | 29.50% |
| LR-4 | 0.7185 | 0.6252 | 1.50 | **0.5277** | **26.56%** |

LR-4가 최악 조건과 상대 하락률에서는 더 안정적이지만 정상 점수가 LR-0보다 0.0456 낮다. 프로젝트 목표가 정상 Macro-F1 최대화이므로 LR-0을 최종 모델로 선택한다.

![LR과 밝기 강건성 비교](docs/figures/efficientnet_lr_comparison.png)

## 11. 결론

“사전학습 본체에는 작은 LR, 새 분류기에는 큰 LR”이라는 일반적인 아이디어가 이번 조건에서는 성능을 높이지 못했다. 마지막 4개 block이 ISIC 병변에 적응하려면 20 epochs에서 `0.0002` 정도의 LR이 필요했고, 본체 LR을 낮추면 underfitting이 나타났다.

따라서 최종 설정은 다음과 같다.

```text
EfficientNet-B0, 320×320
backbone LR = 0.0002
classifier LR = 0.0002
best epoch = 17
dev Macro-F1 = 0.7641
```

## 12. 재현

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_isic2018.ps1 -Connections 8
powershell -ExecutionPolicy Bypass -File scripts\prepare_isic2018.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_efficientnet_lr_experiments.ps1
pytest -q
```

정확한 비교값은 `docs/results/efficientnet_lr_comparison.csv`에 포함한다. 데이터, checkpoint와 전체 예측은 용량 및 라이선스 때문에 Git에서 제외한다.

## 13. 제한

1. 한 번의 seed와 한 dev split 결과이므로 최종 후보는 반복 seed로 확인해야 한다.
2. 다섯 설정을 같은 dev에서 선택했기 때문에 별도 test 성능은 이보다 낮을 수 있다.
3. 밝기 변화는 단순한 전역 배율이며 그림자, 색온도, blur와 압축 노이즈를 포함하지 않는다.
4. ISIC 촬영 장비와 모집단의 편향 때문에 다른 병원이나 스마트폰 영상에 그대로 일반화되지 않을 수 있다.
5. 외부 검증과 전문의 검토 없이 의료 진단에 사용할 수 없다.

## 14. 참고 문헌

- ISIC Challenge, [ISIC 2018 data](https://challenge.isic-archive.com/data/)
- Tschandl et al., *The HAM10000 dataset*, [Scientific Data 5, 180161 (2018)](https://doi.org/10.1038/sdata.2018.161)
- Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
