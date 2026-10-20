import torch

from models.yolo import YOLO
from train.trainer import train_one_epoch, save_checkpoint, load_checkpoint, train

IMG_SIZE = 64  # tiny on purpose -- grids become 8x8, 4x4, 2x2, fast enough for CI
STRIDES = (8, 16, 32)
THRESHOLDS = (16, 32)
NUM_CLASSES = 3


def make_batch(batch_size=2):
    images = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE)
    boxes_list = [
        torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
        torch.tensor([[10.0, 10.0, 15.0, 15.0], [30.0, 30.0, 50.0, 50.0]]),
    ][:batch_size]
    labels_list = [torch.tensor([0]), torch.tensor([1, 2])][:batch_size]
    return images, boxes_list, labels_list


def test_train_one_epoch_runs_and_returns_losses():
    model = YOLO(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = [make_batch()]  # one batch is enough to confirm the loop runs end to end

    stats = train_one_epoch(
        model, dataloader, optimizer, "cpu", IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS
    )

    assert set(stats.keys()) == {"total", "box", "cls"}
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_overfitting_a_single_batch_reduces_loss():
    torch.manual_seed(0)
    model = YOLO(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = make_batch()

    first_loss = None
    last_loss = None
    for _ in range(20):
        stats = train_one_epoch(
            model, [batch], optimizer, "cpu", IMG_SIZE, STRIDES, NUM_CLASSES, THRESHOLDS
        )
        if first_loss is None:
            first_loss = stats["total"]
        last_loss = stats["total"]

    assert last_loss < first_loss  # the model should at least partially memorize one batch


def test_checkpoint_save_and_resume(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoint.pt")

    model_a = YOLO(NUM_CLASSES)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    dataloader = [make_batch()]

    train(
        model_a, dataloader, optimizer_a, "cpu", epochs=2, img_size=IMG_SIZE, strides=STRIDES,
        num_classes=NUM_CLASSES, thresholds=THRESHOLDS, checkpoint_path=checkpoint_path,
    )

    # a fresh model/optimizer picks up exactly where the first one left off
    model_b = YOLO(NUM_CLASSES)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    resumed_epoch = load_checkpoint(checkpoint_path, model_b, optimizer_b, "cpu")

    assert resumed_epoch == 1  # epochs are 0-indexed, 2 epochs ran -> last saved is epoch 1

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b)


def test_train_resumes_without_repeating_completed_epochs(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoint.pt")
    model = YOLO(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = [make_batch()]

    train(
        model, dataloader, optimizer, "cpu", epochs=2, img_size=IMG_SIZE, strides=STRIDES,
        num_classes=NUM_CLASSES, thresholds=THRESHOLDS, checkpoint_path=checkpoint_path,
    )
    # calling train again with the same (now-existing) checkpoint should resume at epoch 2
    # and, since epochs=2 is already satisfied, do nothing -- confirm it doesn't crash either way
    train(
        model, dataloader, optimizer, "cpu", epochs=3, img_size=IMG_SIZE, strides=STRIDES,
        num_classes=NUM_CLASSES, thresholds=THRESHOLDS, checkpoint_path=checkpoint_path,
    )

    resumed_epoch = load_checkpoint(checkpoint_path, model, optimizer, "cpu")
    assert resumed_epoch == 2
