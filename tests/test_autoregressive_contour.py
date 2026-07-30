from __future__ import annotations

import copy

import pytest
import torch

from src.model.autoregressive_contour import AnchoredAutoregressiveContour, BaselineAudioOnly
from src.utils.motion_metrics import autoregressive_loss, masked_position_mse


def make_model(*, balance_decoder_embeddings: bool = False) -> AnchoredAutoregressiveContour:
    torch.manual_seed(7)
    return AnchoredAutoregressiveContour(
        input_dimension=5,
        audio_hidden_dimension=8,
        num_layers=1,
        num_articulators=3,
        output_layer=4,
        embedding_hidden_dimension=16,
        embedding_dimension=6,
        decoder_hidden_dimension=12,
        output_hidden_dimension=16,
        balance_decoder_embeddings=balance_decoder_embeddings,
    )


def sample_batch(device: str = "cpu"):
    torch.manual_seed(11)
    audio = torch.randn(3, 7, 5, device=device)
    targets = torch.randn(3, 7, 3, 4, device=device)
    lengths = torch.tensor([7, 5, 2], device=device)
    return audio, targets, lengths


def test_output_shape_lengths_and_anchor_copy_cpu():
    model = make_model()
    audio, targets, lengths = sample_batch()
    output, _ = model.infer(audio, targets[:, 0], lengths)
    assert output.shape == targets.shape
    assert torch.equal(output[:, 0], targets[:, 0])
    assert torch.equal(output[1, 5:], targets[1, 0].expand(2, -1, -1))
    assert torch.equal(output[2, 2:], targets[2, 0].expand(5, -1, -1))


def test_target_leakage_inference_is_bitwise_identical():
    model = make_model()
    audio, targets, lengths = sample_batch()
    first, _ = model.infer(audio, targets[:, 0], lengths)
    changed = copy.deepcopy(targets)
    changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 1000
    second, _ = model.infer(audio, changed[:, 0], lengths)
    assert torch.equal(first, second)


def test_balanced_embeddings_are_finite_and_shape_preserving():
    model = make_model(balance_decoder_embeddings=True)
    audio, targets, lengths = sample_batch()
    output, diagnostics = model.infer(audio, targets[:, 0], lengths)
    assert output.shape == targets.shape
    assert torch.isfinite(output).all()
    assert diagnostics.audio_embedding_norm > 0
    assert diagnostics.history_embedding_norm > 0


def test_padded_targets_do_not_affect_position_loss():
    prediction = torch.zeros(2, 6, 2, 2)
    target = torch.ones_like(prediction)
    lengths = torch.tensor([6, 3])
    baseline = masked_position_mse(prediction, target, lengths)
    target[1, 3:] = 1e9
    assert torch.equal(baseline, masked_position_mse(prediction, target, lengths))


def test_position_loss_starts_at_frame_one():
    prediction = torch.zeros(1, 4, 1, 2)
    target = torch.zeros_like(prediction)
    target[:, 0] = 1e6
    assert masked_position_mse(prediction, target, torch.tensor([4])).item() == 0.0
    target[:, 1] = 2.0
    assert masked_position_mse(prediction, target, torch.tensor([4])).item() == pytest.approx(
        4.0 / 3.0
    )


def test_teacher_forcing_target_shift_uses_previous_frame():
    model = make_model()
    audio, targets, lengths = sample_batch()
    observed: list[torch.Tensor] = []

    def hook(_module, inputs):
        observed.append(inputs[0].detach().clone())

    handle = model.history_encoder.register_forward_pre_hook(hook)
    try:
        model(
            audio,
            lengths,
            targets[:, 0],
            targets=targets,
            teacher_forcing_ratio=1.0,
        )
    finally:
        handle.remove()
    anchor = targets[:, 0].reshape(3, -1)
    assert torch.equal(observed[0], targets[:, 0].reshape(3, -1) - anchor)
    assert torch.equal(observed[1], targets[:, 1].reshape(3, -1) - anchor)


def test_loss_backward_connects_history_encoder():
    model = make_model()
    audio, targets, lengths = sample_batch()
    prediction, _ = model(
        audio,
        lengths,
        targets[:, 0],
        targets=targets,
        teacher_forcing_ratio=1.0,
    )
    losses = autoregressive_loss(
        prediction,
        targets,
        lengths,
        velocity_weight=0.5,
        class_weights=torch.ones(3),
    )
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.history_encoder.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_audio_evaluation_wrapper_matches_untouched_baseline():
    from src.model.baseline_5 import BaselineModel

    torch.manual_seed(19)
    baseline = BaselineModel(5, 8, 1, 4, 3, 1, "unused").eval()
    wrapper = BaselineAudioOnly(
        {
            "input_layer": 5,
            "hidden_layer": 8,
            "num_layers": 1,
            "output_layer": 4,
            "classes": ["a", "b", "c"],
        }
    ).eval()
    wrapper.audio_encoder.load_state_dict(
        {
            key: baseline.state_dict()[key]
            for key in wrapper.audio_encoder.state_dict()
        },
        strict=True,
    )
    wrapper.readout_layer.load_state_dict(baseline.readout_layer.state_dict(), strict=True)
    audio, _, lengths = sample_batch()
    expected, _, _ = baseline(audio, lengths)
    actual = wrapper(audio, lengths)
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward():
    model = make_model().cuda()
    audio, targets, lengths = sample_batch("cuda")
    output, _ = model.infer(audio, targets[:, 0], lengths)
    assert output.is_cuda
    assert output.shape == targets.shape
