import torch


def _main_model_only_state_keys(model):
    """Return target-architecture state that must remain from the main model."""
    if model.__class__.__name__ == "ViCLIP":
        return {
            name
            for name in model.state_dict()
            if "temporal_positional_embedding" in name
        }
    return set()


def merge_model_weights_(model, wiseft_model, coef=0.5):
    """Interpolate a main model with a compatible additional model in place.

    Shared state is interpolated. State added by a derived architecture, such
    as ViCLIP temporal embeddings, remains unchanged from the main ``model``
    because it has no meaningful counterpart in the additional checkpoint
    used for interpolation.
    """
    if not 0.0 <= coef <= 1.0:
        raise ValueError(f"wiseft_coef must be between 0 and 1, got {coef}")

    main_state = model.state_dict()
    wiseft_state = wiseft_model.state_dict()
    main_keys = set(main_state)
    wiseft_keys = set(wiseft_state)
    if main_keys != wiseft_keys:
        missing = sorted(main_keys - wiseft_keys)
        unexpected = sorted(wiseft_keys - main_keys)
        raise ValueError(
            "WiSE-FT checkpoint is incompatible with the main model: "
            f"missing keys={missing[:10]}, unexpected keys={unexpected[:10]}"
        )

    main_model_only_keys = _main_model_only_state_keys(model)
    with torch.no_grad():
        for name, main_tensor in main_state.items():
            wiseft_tensor = wiseft_state[name]
            if main_tensor.shape != wiseft_tensor.shape:
                raise ValueError(
                    f"WiSE-FT tensor shape mismatch for {name!r}: "
                    f"{tuple(main_tensor.shape)} != {tuple(wiseft_tensor.shape)}"
                )
            wiseft_tensor = wiseft_tensor.to(
                device=main_tensor.device,
                dtype=main_tensor.dtype,
            )
            if (
                name not in main_model_only_keys
                and (
                    torch.is_floating_point(main_tensor)
                    or torch.is_complex(main_tensor)
                )
            ):
                main_tensor.lerp_(
                    wiseft_tensor,
                    coef,
                )
            else:
                print(name)

    return model
