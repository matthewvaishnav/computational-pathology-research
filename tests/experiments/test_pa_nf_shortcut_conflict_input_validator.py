from scripts.paired_acquisition.validate_pa_nf_shortcut_conflict_inputs import main


def test_input_validator_entrypoint_is_importable() -> None:
    assert callable(main)
