import torch

from src.models.wsi_nca import WSINCA, build_neighbor_index


def test_spatial_knn_uses_slide_coordinates():
    states = torch.zeros(1, 4, 3)
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [12.0, 0.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    neighbors = build_neighbor_index(states, coordinates, mask, k=1, mode="spatial")

    assert neighbors.shape == (1, 4, 1)
    assert neighbors[0, 0, 0].item() == 1
    assert neighbors[0, 1, 0].item() == 0
    assert neighbors[0, 2, 0].item() == 3
    assert neighbors[0, 3, 0].item() == 2


def test_forward_exposes_states_and_ignores_padding():
    torch.manual_seed(7)
    model = WSINCA(
        input_dim=8,
        hidden_dim=16,
        num_classes=6,
        num_steps=2,
        k_neighbors=2,
        dropout=0.0,
    ).eval()

    features = torch.randn(2, 5, 8)
    coordinates = torch.tensor(
        [
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            [[0, 0], [1, 0], [2, 0], [99, 99], [100, 100]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ],
    )

    output = model(features, coordinates, mask)

    assert output.logits.shape == (2, 6)
    assert output.slide_state.shape == (2, 16)
    assert output.cell_state.shape == (2, 5, 16)
    assert output.neighbor_index.shape == (2, 5, 2)
    assert torch.count_nonzero(output.cell_state[1, 3:]).item() == 0


def test_t0_is_coordinate_blind_static_control():
    torch.manual_seed(11)
    model = WSINCA(
        input_dim=4,
        hidden_dim=8,
        num_classes=3,
        num_steps=0,
        k_neighbors=2,
        dropout=0.0,
    ).eval()

    features = torch.randn(1, 4, 4)
    coordinates_a = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]]])
    coordinates_b = torch.tensor(
        [[[100.0, 50.0], [-50.0, 9.0], [1.0, 1000.0], [500.0, -200.0]]],
    )

    output_a = model(features, coordinates_a)
    output_b = model(features, coordinates_b)

    assert output_a.neighbor_index.shape == (1, 4, 0)
    assert output_b.neighbor_index.shape == (1, 4, 0)
    torch.testing.assert_close(output_a.logits, output_b.logits, rtol=0.0, atol=0.0)


def test_spatial_topology_changes_when_coordinates_are_reassigned():
    states = torch.zeros(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]])
    reassigned = coordinates[:, torch.tensor([0, 2, 1, 3])]

    original = build_neighbor_index(states, coordinates, mask, k=1, mode="spatial")
    shuffled = build_neighbor_index(states, reassigned, mask, k=1, mode="spatial")

    assert not torch.equal(original, shuffled)


def test_tied_parameter_count_does_not_grow_with_developmental_steps():
    model_t1 = WSINCA(input_dim=8, hidden_dim=16, num_steps=1, dynamics_mode="tied")
    model_t16 = WSINCA(input_dim=8, hidden_dim=16, num_steps=16, dynamics_mode="tied")

    count_t1 = sum(parameter.numel() for parameter in model_t1.parameters())
    count_t16 = sum(parameter.numel() for parameter in model_t16.parameters())

    assert count_t1 == count_t16
    assert set(model_t1.state_dict()) == set(model_t16.state_dict())


def test_untied_gnn_control_has_more_parameters_at_equal_width():
    tied = WSINCA(input_dim=8, hidden_dim=16, num_steps=4, dynamics_mode="tied")
    untied = WSINCA(input_dim=8, hidden_dim=16, num_steps=4, dynamics_mode="untied")

    tied_count = sum(parameter.numel() for parameter in tied.parameters())
    untied_count = sum(parameter.numel() for parameter in untied.parameters())

    assert untied_count > tied_count


def test_joint_patch_permutation_preserves_slide_prediction():
    torch.manual_seed(19)
    model = WSINCA(
        input_dim=6,
        hidden_dim=12,
        num_classes=4,
        num_steps=3,
        k_neighbors=2,
        dropout=0.0,
    ).eval()

    features = torch.randn(1, 5, 6)
    coordinates = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [10.0, 0.0], [20.0, 0.0]]],
    )
    mask = torch.ones(1, 5, dtype=torch.bool)

    reference = model(features, coordinates, mask).logits

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        features[:, permutation],
        coordinates[:, permutation],
        mask[:, permutation],
    ).logits

    torch.testing.assert_close(reference, permuted, rtol=1e-5, atol=1e-6)


def test_relative_position_dynamics_are_translation_invariant():
    torch.manual_seed(23)
    model = WSINCA(
        input_dim=5,
        hidden_dim=10,
        num_classes=2,
        num_steps=2,
        k_neighbors=2,
        dropout=0.0,
    ).eval()

    features = torch.randn(1, 6, 5)
    coordinates = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]]
    )
    translated = coordinates + torch.tensor([[[137.0, -89.0]]])

    reference = model(features, coordinates)
    shifted = model(features, translated)

    assert torch.equal(reference.neighbor_index, shifted.neighbor_index)
    torch.testing.assert_close(reference.cell_state, shifted.cell_state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reference.logits, shifted.logits, rtol=0.0, atol=0.0)
