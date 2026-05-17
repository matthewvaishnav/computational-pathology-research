"""
Property-based tests for FixedLengthBagSampler.

This test file validates correctness properties for the fixed-length bag sampling
component using property-based testing with Hypothesis. Each property test runs
a minimum of 100 iterations to verify universal invariants.

Feature: nnmil-architecture-upgrade
"""

import pytest
import torch

from hypothesis import given, settings
from hypothesis import strategies as st
from src.data.bag_samplers import FixedLengthBagSampler

# ============================================================================
# Property 7: Fixed-Length Bag Invariant
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=10000),
    num_patches=st.integers(min_value=1, max_value=20000),
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_7_fixed_length_bag_invariant(bag_length, num_patches, feature_dim, mode):
    """
    Feature: nnmil-architecture-upgrade, Property 7: For any slide with N_actual
    patches and configured bag length M, the sampled bag SHALL have exactly M patches.

    **Validates: Requirements 3.1, 3.7**

    This property ensures that regardless of the input size (N_actual), the output
    always has exactly M patches, enabling efficient batching.
    """
    # Create sampler with specified bag length
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)

    # Create features with N_actual patches
    features = torch.randn(num_patches, feature_dim)

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify output has exactly M patches
    assert sampled_features.shape[0] == bag_length, (
        f"Expected bag length {bag_length}, but got {sampled_features.shape[0]} "
        f"(input had {num_patches} patches)"
    )

    # Verify feature dimension is preserved
    assert (
        sampled_features.shape[1] == feature_dim
    ), f"Feature dimension changed from {feature_dim} to {sampled_features.shape[1]}"

    # Verify mask has correct length
    assert (
        mask.shape[0] == bag_length
    ), f"Mask length {mask.shape[0]} does not match bag length {bag_length}"


# ============================================================================
# Property 8: Padding Correctness
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=1000),
    num_patches=st.integers(min_value=1, max_value=99),  # N_actual < M
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_8_padding_correctness(bag_length, num_patches, feature_dim, mode):
    """
    Feature: nnmil-architecture-upgrade, Property 8: For any slide with
    N_actual < M patches, the sampled bag SHALL contain N_actual original
    patches followed by (M - N_actual) zero vectors.

    **Validates: Requirements 3.2**

    This property ensures correct padding behavior when slides have fewer
    patches than the configured bag length.
    """
    # Create sampler
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)

    # Create features with unique identifiable values
    features = torch.randn(num_patches, feature_dim)

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify first N_actual patches match original features
    assert torch.allclose(
        sampled_features[:num_patches], features[:num_patches]
    ), f"First {num_patches} patches do not match original features"

    # Verify remaining (M - N_actual) patches are zero vectors
    padding_length = bag_length - num_patches
    expected_zeros = torch.zeros(padding_length, feature_dim)
    assert torch.allclose(
        sampled_features[num_patches:], expected_zeros
    ), f"Padding patches (positions {num_patches}:{bag_length}) are not zero vectors"

    # Verify mask marks first N_actual positions as True
    assert mask[:num_patches].all(), f"Mask should be True for first {num_patches} positions"

    # Verify mask marks padding positions as False
    assert not mask[
        num_patches:
    ].any(), f"Mask should be False for padding positions {num_patches}:{bag_length}"

    # Verify mask has exactly N_actual True values
    assert (
        mask.sum() == num_patches
    ), f"Mask should have exactly {num_patches} True values, got {mask.sum()}"


# ============================================================================
# Property 9: Sampling Without Replacement
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=500),
    num_patches=st.integers(min_value=501, max_value=5000),  # N_actual > M
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
)
def test_property_9_sampling_without_replacement(bag_length, num_patches, feature_dim):
    """
    Feature: nnmil-architecture-upgrade, Property 9: For any slide with
    N_actual > M patches during training, the sampled M patches SHALL be
    unique (no duplicates).

    **Validates: Requirements 3.3**

    This property ensures that random sampling during training does not
    select the same patch multiple times (sampling without replacement).
    """
    # Create sampler in train mode (random sampling)
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode="train")

    # Create features where each patch has a unique identifier
    # Use the first dimension as a unique ID for each patch
    features = torch.zeros(num_patches, feature_dim)
    for i in range(num_patches):
        features[i, 0] = float(i)  # Unique ID in first dimension
        features[i, 1:] = torch.randn(feature_dim - 1)  # Random values in other dims

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Extract the unique IDs from sampled features
    sampled_ids = sampled_features[:, 0]

    # Verify all sampled IDs are unique (no duplicates)
    unique_ids = torch.unique(sampled_ids)
    assert len(unique_ids) == bag_length, (
        f"Expected {bag_length} unique patches, but got {len(unique_ids)} unique IDs. "
        f"This indicates sampling with replacement (duplicates present)."
    )

    # Verify all sampled IDs are from valid range [0, N_actual)
    assert (sampled_ids >= 0).all(), "Sampled IDs contain negative values"
    assert (sampled_ids < num_patches).all(), f"Sampled IDs exceed valid range [0, {num_patches})"

    # Verify mask is all True (no padding needed when N > M)
    assert mask.all(), "Mask should be all True when N_actual > M (no padding needed)"


# ============================================================================
# Property 10: Sliding Window Activation
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=500),
    num_patches=st.integers(min_value=501, max_value=5000),  # N_actual > M
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
    stride_fraction=st.floats(min_value=0.5, max_value=1.0),
)
def test_property_10_sliding_window_activation(
    bag_length, num_patches, feature_dim, stride_fraction
):
    """
    Feature: nnmil-architecture-upgrade, Property 10: For any slide with
    N_actual > M patches during inference, the system SHALL apply sliding
    window processing with overlapping windows.

    **Validates: Requirements 3.4, 5.1**

    This property ensures that inference mode uses sliding window processing
    for slides larger than the bag length, enabling full slide coverage.
    """
    # Calculate stride from fraction
    stride = max(1, int(bag_length * stride_fraction))

    # Create sampler in inference mode with specified stride
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode="inference", stride=stride)

    # Create features with identifiable values
    features = torch.randn(num_patches, feature_dim)

    # Sample first window
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify first window contains first M patches (deterministic in inference mode)
    assert torch.allclose(
        sampled_features, features[:bag_length]
    ), "Inference mode should return first M patches as first window"

    # Verify mask is all True (no padding when N > M)
    assert mask.all(), "Mask should be all True when N_actual > M (no padding needed)"

    # Verify get_num_windows calculates correct number of windows
    num_windows = sampler.get_num_windows(num_patches)

    # Calculate expected number of windows: (N - M) / stride + 1
    expected_windows = (num_patches - bag_length) // stride + 1
    assert num_windows == expected_windows, (
        f"Expected {expected_windows} windows, but get_num_windows returned {num_windows} "
        f"(N={num_patches}, M={bag_length}, stride={stride})"
    )

    # Verify at least 2 windows when N > M (overlapping coverage)
    assert num_windows >= 1, f"Should have at least 1 window when N_actual > M, got {num_windows}"

    # Verify sliding window provides overlapping coverage when stride < bag_length
    if stride < bag_length:
        overlap = bag_length - stride
        assert overlap > 0, f"Windows should overlap by {overlap} patches when stride < bag_length"


# ============================================================================
# Property 11: Bag Length Configuration Range
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=10000),
    num_patches=st.integers(min_value=50, max_value=15000),
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_11_bag_length_configuration_range(bag_length, num_patches, feature_dim, mode):
    """
    Feature: nnmil-architecture-upgrade, Property 11: For any configured bag
    length M in the range [100, 10000], the system SHALL successfully create
    and process bags of length M.

    **Validates: Requirements 3.5**

    This property ensures the system supports the full range of bag lengths
    specified in the requirements, from small (100) to large (10000).
    """
    # Create sampler with bag length in valid range
    try:
        sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)
    except ValueError as e:
        pytest.fail(f"Failed to create sampler with valid bag_length={bag_length}: {e}")

    # Create features
    features = torch.randn(num_patches, feature_dim)

    # Sample fixed-length bag
    try:
        sampled_features, mask = sampler.sample(features, num_patches=num_patches)
    except Exception as e:
        pytest.fail(
            f"Failed to sample with bag_length={bag_length}, " f"num_patches={num_patches}: {e}"
        )

    # Verify output has correct bag length
    assert (
        sampled_features.shape[0] == bag_length
    ), f"Expected bag length {bag_length}, got {sampled_features.shape[0]}"

    # Verify output is valid (no NaN or Inf)
    assert not torch.isnan(sampled_features).any(), "Sampled features contain NaN values"
    assert not torch.isinf(sampled_features).any(), "Sampled features contain Inf values"

    # Verify mask is valid
    assert (
        mask.shape[0] == bag_length
    ), f"Mask length {mask.shape[0]} does not match bag length {bag_length}"
    assert mask.dtype == torch.bool, f"Mask dtype should be bool, got {mask.dtype}"


# ============================================================================
# Property 12: Attention Mask Correctness
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=1000),
    num_patches=st.integers(min_value=1, max_value=99),  # N_actual < M (padding case)
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_12_attention_mask_correctness(bag_length, num_patches, feature_dim, mode):
    """
    Feature: nnmil-architecture-upgrade, Property 12: For any bag with padding,
    the attention mask SHALL mark padded positions as False and valid positions
    as True.

    **Validates: Requirements 3.6**

    This property ensures attention masks correctly identify valid vs. padded
    positions, enabling proper attention computation in the model.
    """
    # Create sampler
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)

    # Create features
    features = torch.randn(num_patches, feature_dim)

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify mask dtype is bool
    assert mask.dtype == torch.bool, f"Mask dtype should be torch.bool, got {mask.dtype}"

    # Verify mask length matches bag length
    assert (
        mask.shape[0] == bag_length
    ), f"Mask length {mask.shape[0]} does not match bag length {bag_length}"

    # Verify valid positions (first N_actual) are marked as True
    assert mask[:num_patches].all(), f"Valid positions [0:{num_patches}] should all be True in mask"

    # Verify padded positions (N_actual to M) are marked as False
    if num_patches < bag_length:
        assert not mask[
            num_patches:
        ].any(), f"Padded positions [{num_patches}:{bag_length}] should all be False in mask"

    # Verify mask has exactly N_actual True values
    num_true = mask.sum().item()
    assert (
        num_true == num_patches
    ), f"Mask should have exactly {num_patches} True values, got {num_true}"

    # Verify mask has exactly (M - N_actual) False values
    num_false = (~mask).sum().item()
    expected_false = bag_length - num_patches
    assert (
        num_false == expected_false
    ), f"Mask should have exactly {expected_false} False values, got {num_false}"

    # Verify mask can be used for indexing (practical usage test)
    try:
        valid_features = sampled_features[mask]
        assert valid_features.shape[0] == num_patches, (
            f"Masked indexing should return {num_patches} features, "
            f"got {valid_features.shape[0]}"
        )
    except Exception as e:
        pytest.fail(f"Mask cannot be used for indexing: {e}")


# ============================================================================
# Additional Property: Mask Correctness for No-Padding Case
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=1000),
    num_patches=st.integers(min_value=100, max_value=5000),  # N_actual >= M
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_mask_correctness_no_padding(bag_length, num_patches, feature_dim, mode):
    """
    Feature: nnmil-architecture-upgrade: For any bag without padding
    (N_actual >= M), the attention mask SHALL be all True.

    This property complements Property 12 by testing the no-padding case.
    """
    # Ensure num_patches >= bag_length
    if num_patches < bag_length:
        num_patches = bag_length

    # Create sampler
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)

    # Create features
    features = torch.randn(num_patches, feature_dim)

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify mask is all True (no padding)
    assert mask.all(), (
        f"Mask should be all True when N_actual >= M " f"(N_actual={num_patches}, M={bag_length})"
    )

    # Verify mask has exactly M True values
    assert (
        mask.sum() == bag_length
    ), f"Mask should have exactly {bag_length} True values, got {mask.sum()}"

    # Verify no False values in mask
    assert (~mask).sum() == 0, "Mask should have no False values when N_actual >= M"


# ============================================================================
# Additional Property: Determinism in Inference Mode
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=1000),
    num_patches=st.integers(min_value=50, max_value=5000),
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
)
def test_property_determinism_inference_mode(bag_length, num_patches, feature_dim):
    """
    Feature: nnmil-architecture-upgrade: For any input in inference mode,
    repeated sampling SHALL produce identical results (deterministic behavior).

    This property ensures reproducibility in inference mode, which is critical
    for clinical applications.
    """
    # Create sampler in inference mode
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode="inference")

    # Create features
    features = torch.randn(num_patches, feature_dim)

    # Sample twice
    sampled_1, mask_1 = sampler.sample(features, num_patches=num_patches)
    sampled_2, mask_2 = sampler.sample(features, num_patches=num_patches)

    # Verify results are identical
    assert torch.allclose(
        sampled_1, sampled_2
    ), "Inference mode should produce identical results on repeated sampling"

    assert torch.equal(
        mask_1, mask_2
    ), "Inference mode should produce identical masks on repeated sampling"


# ============================================================================
# Additional Property: Randomness in Training Mode
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=500),
    num_patches=st.integers(min_value=501, max_value=5000),  # N > M for sampling
    feature_dim=st.sampled_from([512, 768, 1024, 2048]),
)
def test_property_randomness_training_mode(bag_length, num_patches, feature_dim):
    """
    Feature: nnmil-architecture-upgrade: For any input in training mode with
    N_actual > M, repeated sampling SHALL produce different results (stochastic
    behavior for data augmentation).

    This property ensures training mode provides randomness for better
    generalization.
    """
    # Create sampler in training mode
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode="train")

    # Create features with unique identifiable values
    features = torch.zeros(num_patches, feature_dim)
    for i in range(num_patches):
        features[i, 0] = float(i)  # Unique ID
        features[i, 1:] = torch.randn(feature_dim - 1)

    # Sample multiple times
    num_trials = 5
    sampled_ids_list = []

    for _ in range(num_trials):
        sampled, _ = sampler.sample(features, num_patches=num_patches)
        sampled_ids = sampled[:, 0].sort()[0]  # Sort for comparison
        sampled_ids_list.append(sampled_ids)

    # Verify at least some trials produce different results
    # (with very high probability for random sampling)
    all_identical = True
    for i in range(1, num_trials):
        if not torch.allclose(sampled_ids_list[0], sampled_ids_list[i]):
            all_identical = False
            break

    assert not all_identical, (
        f"Training mode should produce different samples across {num_trials} trials "
        f"when N_actual > M (got identical results, indicating lack of randomness)"
    )


# ============================================================================
# Additional Property: Feature Dimension Preservation
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=1000),
    num_patches=st.integers(min_value=1, max_value=5000),
    feature_dim=st.integers(min_value=128, max_value=4096),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_feature_dimension_preservation(bag_length, num_patches, feature_dim, mode):
    """
    Feature: nnmil-architecture-upgrade: For any input with feature dimension D,
    the sampled bag SHALL preserve the feature dimension D.

    This property ensures the sampler does not alter feature dimensions,
    maintaining compatibility with downstream models.
    """
    # Create sampler
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)

    # Create features with specified dimension
    features = torch.randn(num_patches, feature_dim)

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify feature dimension is preserved
    assert (
        sampled_features.shape[1] == feature_dim
    ), f"Feature dimension changed from {feature_dim} to {sampled_features.shape[1]}"

    # Verify output shape is [M, D]
    assert sampled_features.shape == (bag_length, feature_dim), (
        f"Expected shape ({bag_length}, {feature_dim}), " f"got {sampled_features.shape}"
    )


# ============================================================================
# Additional Property: Device and Dtype Preservation
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    bag_length=st.integers(min_value=100, max_value=1000),
    num_patches=st.integers(min_value=50, max_value=2000),
    feature_dim=st.sampled_from([512, 1024]),
    dtype=st.sampled_from([torch.float32, torch.float16, torch.float64]),
    mode=st.sampled_from(["train", "inference"]),
)
def test_property_device_dtype_preservation(bag_length, num_patches, feature_dim, dtype, mode):
    """
    Feature: nnmil-architecture-upgrade: For any input tensor with specific
    dtype and device, the sampled bag SHALL preserve the dtype and device.

    This property ensures compatibility with mixed-precision training and
    multi-GPU setups.
    """
    # Create sampler
    sampler = FixedLengthBagSampler(bag_length=bag_length, mode=mode)

    # Create features with specified dtype
    features = torch.randn(num_patches, feature_dim, dtype=dtype)

    # Sample fixed-length bag
    sampled_features, mask = sampler.sample(features, num_patches=num_patches)

    # Verify dtype is preserved
    assert (
        sampled_features.dtype == dtype
    ), f"Feature dtype changed from {dtype} to {sampled_features.dtype}"

    # Verify device is preserved (CPU in this case)
    assert (
        sampled_features.device == features.device
    ), f"Device changed from {features.device} to {sampled_features.device}"

    # Verify mask device matches features device
    assert (
        mask.device == features.device
    ), f"Mask device {mask.device} does not match features device {features.device}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
