import numpy as np

from skin_project.checkpoint import save_checkpoint
from skin_project.inference import run_inference
from skin_project.models import EfficientNetB0


CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def _checkpoint(tmp_path):
    model = EfficientNetB0(pretrained=False, freeze_encoder=True)
    config = {
        "type": "efficientnet_b0",
        "backbone": "efficientnet_b0",
        "num_classes": 7,
        "pretrained": False,
        "freeze_encoder": True,
    }
    return save_checkpoint(tmp_path / "model.pt", model, config, 64, 0, CLASSES)


def test_inference_exports_all_seven_probabilities(tiny_dataset, tmp_path):
    result = run_inference(
        _checkpoint(tmp_path),
        tiny_dataset["root"] / "dev_metadata.csv",
        tiny_dataset["root"] / "dev_images",
        tmp_path / "predictions.csv",
        device="cpu",
        image_ext=".jpg",
    )
    probability_columns = [column for column in result if column.startswith("probability_")]
    assert probability_columns == [f"probability_{name}" for name in CLASSES]
    assert np.allclose(result[probability_columns].sum(axis=1), 1.0)
    assert set(result["predicted_label"]) <= set(CLASSES)
