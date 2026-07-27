import pytest
import torch
from argparse import Namespace

from models import ACMG, BaselineAT
from models.acmg import CrossModalGate
from utils import TrainTool_baseline, TrainTool_baseline_2class


@pytest.mark.parametrize("model_class", [BaselineAT, ACMG])
def test_model_forward(model_class):
    model = model_class(
        input_dim=8,
        ffn_embed_dim=16,
        num_layers=1,
        num_heads=2,
        num_classes=3,
    ).eval()

    audio = torch.randn(2, 7, 8)
    text = torch.randn(2, 5, 8)
    audio_mask = torch.zeros(2, 7, dtype=torch.bool)
    text_mask = torch.zeros(2, 5, dtype=torch.bool)
    audio_mask[0, 5:] = True
    text_mask[1, 3:] = True

    with torch.inference_mode():
        output = model(audio, text, audio_mask, text_mask)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_cross_modal_gate_shapes_and_padding():
    gate = CrossModalGate(dim=8).eval()
    audio = torch.randn(2, 7, 8)
    text = torch.randn(2, 5, 8)
    audio_mask = torch.zeros(2, 7, dtype=torch.bool)
    text_mask = torch.zeros(2, 5, dtype=torch.bool)
    audio_mask[0, 5:] = True
    text_mask[1, 3:] = True

    with torch.inference_mode():
        gated_audio, gated_text, audio_gate, text_gate = gate(
            audio, text, audio_mask, text_mask
        )

    assert gated_audio.shape == audio.shape
    assert gated_text.shape == text.shape
    assert audio_gate.shape == (2, 7, 1)
    assert text_gate.shape == (2, 5, 1)
    assert torch.count_nonzero(audio_gate[0, 5:]) == 0
    assert torch.count_nonzero(text_gate[1, 3:]) == 0


@pytest.mark.parametrize("model_type, expected_type", [
    ("A_T", BaselineAT),
    ("GateModel_A_T_bytime", ACMG),
])
@pytest.mark.parametrize("training_tool,num_classes", [
    (TrainTool_baseline, 3),
    (TrainTool_baseline_2class, 2),
])
def test_training_model_registry(model_type, expected_type, training_tool, num_classes):
    config = Namespace(
        model_type=model_type,
        model_config={model_type: {"num_layers": 1, "num_heads": 2}},
        num_classes=num_classes,
        input_dim=8,
        ffn_embed_dim=16,
        project_path=".",
        device="cpu",
    )

    assert isinstance(training_tool.get_model(config), expected_type)
