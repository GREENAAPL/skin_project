from skin_project.dataloaders import build_dataloader, build_dataloaders_from_config
from skin_project.dataset import SkinLesionDataset


def test_batch_schema(tiny_dataset):
    dataset = SkinLesionDataset(
        tiny_dataset["train_df"],
        tiny_dataset["root"] / "train_images",
        image_size=64,
        image_ext=".jpg",
    )
    batch = next(iter(build_dataloader(dataset, batch_size=4, shuffle=False)))
    assert set(batch) == {"image", "label", "sample_id", "lesion_id"}
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["label"].shape == (4,)


def test_build_from_config_is_image_only(tiny_config):
    bundle = build_dataloaders_from_config(tiny_config)
    batch = next(iter(bundle.train_loader))
    assert "metadata" not in batch
    assert batch["image"].ndim == 4
