import pytest
import torch
from torch import nn

from clip_benchmark.models.wise_ft import merge_model_weights_


class TinyModel(nn.Module):
    def __init__(self, value, counter):
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), value))
        self.register_buffer("running_value", torch.tensor(float(value)))
        self.register_buffer("counter", torch.tensor(counter, dtype=torch.long))


def test_merge_model_weights_interpolates_floating_tensors_in_place():
    base = TinyModel(value=0.0, counter=1)
    fine_tuned = TinyModel(value=4.0, counter=9)

    result = merge_model_weights_(base, fine_tuned, coef=0.25)

    assert result is base
    torch.testing.assert_close(base.weight, torch.ones(2))
    torch.testing.assert_close(base.running_value, torch.tensor(1.0))
    assert base.counter.item() == 1


def test_merge_model_weights_rejects_out_of_range_coefficient():
    with pytest.raises(ValueError, match="between 0 and 1"):
        merge_model_weights_(TinyModel(0.0, 0), TinyModel(1.0, 0), coef=1.1)


def test_merge_model_weights_rejects_incompatible_models():
    base = nn.Linear(2, 2)
    fine_tuned = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="shape mismatch"):
        merge_model_weights_(base, fine_tuned)


class ViCLIP(nn.Module):
    def __init__(self, spatial_value, temporal_value):
        super().__init__()
        self.spatial_embedding = nn.Parameter(torch.tensor(float(spatial_value)))
        self.temporal_positional_embedding = nn.Parameter(
            torch.tensor(float(temporal_value))
        )


def test_viclip_keeps_main_model_temporal_embeddings_uninterpolated():
    trained_viclip = ViCLIP(spatial_value=4.0, temporal_value=8.0)
    datacomp_initialized = ViCLIP(spatial_value=0.0, temporal_value=0.0)

    merge_model_weights_(trained_viclip, datacomp_initialized, coef=0.25)

    torch.testing.assert_close(
        trained_viclip.spatial_embedding,
        torch.tensor(3.0),
    )
    torch.testing.assert_close(
        trained_viclip.temporal_positional_embedding,
        torch.tensor(8.0),
    )
