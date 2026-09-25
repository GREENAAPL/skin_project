import torch

from skin_project.dataset import SkinLesionDataset


def _dataset(tiny_dataset, frame=None, **kwargs):
    return SkinLesionDataset(
        frame if frame is not None else tiny_dataset["train_df"],
        tiny_dataset["root"] / "train_images",
        image_size=64,
        image_ext=".jpg",
        **kwargs,
    )


def test_image_only_sample_schema(tiny_dataset):
    sample = _dataset(tiny_dataset)[0]
    assert set(sample) == {"image", "label", "sample_id", "lesion_id"}
    assert sample["image"].shape == (3, 64, 64)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.long


def test_sample_id_controls_image_lookup_after_row_shuffle(tiny_dataset):
    frame = tiny_dataset["train_df"]
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    original = _dataset(tiny_dataset, frame=frame)
    reordered = _dataset(tiny_dataset, frame=shuffled)
    target = frame.iloc[3]["sample_id"]
    target_index = int(shuffled.index[shuffled["sample_id"] == target][0])
    assert original[3]["sample_id"] == reordered[target_index]["sample_id"]
    assert torch.allclose(original[3]["image"], reordered[target_index]["image"])


def test_unlabeled_dataset_emits_minus_one(tiny_dataset):
    assert int(_dataset(tiny_dataset, has_labels=False)[0]["label"]) == -1
