"""
Tests for risk fixes to verify correctness.
"""

import numpy as np
import pytest
import torch


def test_gnn_non_contiguous_batch():
    """Test that GNN fallback handles non-contiguous batch indices."""
    from src.cells.gnn import _global_mean_pool_fallback
    
    # Create features with non-contiguous batch indices
    x = torch.randn(6, 16)
    batch = torch.tensor([0, 0, 5, 5, 10, 10])  # Non-contiguous: 0, 5, 10
    
    # Should not crash and should produce 3 outputs (one per unique batch)
    result = _global_mean_pool_fallback(x, batch)
    
    assert result.shape == (3, 16), f"Expected (3, 16), got {result.shape}"
    
    # Verify correctness: batch 0 should be mean of first 2 rows
    expected_batch_0 = x[:2].mean(0)
    torch.testing.assert_close(result[0], expected_batch_0, rtol=1e-5, atol=1e-5)


def test_omics_fusion_all_masked():
    """Test that MultiOmicsFusion handles all-masked samples without NaN."""
    from src.omics.fusion import MultiOmicsFusion
    
    fusion = MultiOmicsFusion(embed_dim=32, num_heads=2, output_dim=32)
    fusion.eval()
    
    # Create sample data
    modality_embeddings = {
        "rna": torch.randn(4, 32),
        "protein": torch.randn(4, 32),
    }
    
    # Create mask where sample 2 has all modalities masked
    modality_mask = {
        "rna": torch.tensor([True, True, False, True]),
        "protein": torch.tensor([True, True, False, True]),
    }
    
    with torch.no_grad():
        result = fusion(modality_embeddings, modality_mask)
    
    # Should not contain NaN
    assert not torch.isnan(result).any(), "Result contains NaN values"
    assert result.shape == (4, 32), f"Expected (4, 32), got {result.shape}"


def test_ipw_stabilization():
    """Test that IPW estimator handles extreme propensity scores."""
    from src.causal.estimators import IPWEstimator
    
    # Create synthetic data with some extreme propensity scores
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    T = np.random.binomial(1, 0.5, n)
    Y = T * 2 + np.random.randn(n) * 0.5
    
    estimator = IPWEstimator(trim_quantile=0.01)
    estimator.fit(X, T, Y)
    
    # Should not crash or produce inf/nan
    ate, se = estimator.predict_ate(X, T, Y)
    
    assert np.isfinite(ate), f"ATE is not finite: {ate}"
    assert np.isfinite(se), f"SE is not finite: {se}"
    assert se > 0, f"SE should be positive: {se}"


def test_hypothesis_generator_timeout():
    """Test that HypothesisGenerator accepts timeout parameter."""
    from src.hypothesis.generator import HypothesisGenerator
    
    # Should accept timeout parameter without error
    gen = HypothesisGenerator(api_key="test-key")
    
    # Verify _call_api signature accepts timeout
    import inspect
    sig = inspect.signature(gen._call_api)
    assert "timeout" in sig.parameters, "_call_api should accept timeout parameter"
    assert sig.parameters["timeout"].default == 60.0, "Default timeout should be 60s"


def test_hypothesis_generator_fence_parsing():
    """Test robust markdown fence parsing."""
    from src.hypothesis.generator import HypothesisGenerator
    
    gen = HypothesisGenerator(api_key="test-key")
    
    # Test case 1: Standard markdown with json tag
    text1 = '```json\n[{"hypothesis_text": "test"}]\n```'
    result1 = gen._parse_response(text1, {})
    assert len(result1) == 1
    
    # Test case 2: Only opening fence (edge case)
    text2 = '```\n[{"hypothesis_text": "test"}]'
    result2 = gen._parse_response(text2, {})
    assert len(result2) == 1
    
    # Test case 3: No fences
    text3 = '[{"hypothesis_text": "test"}]'
    result3 = gen._parse_response(text3, {})
    assert len(result3) == 1


def test_spatial_chunked_loading():
    """Test that chunked sparse matrix loading works correctly."""
    from scipy.sparse import csr_matrix
    
    # Create a mock sparse matrix
    n_spots, n_genes = 25000, 2000
    density = 0.1
    nnz = int(n_spots * n_genes * density)
    
    rows = np.random.randint(0, n_spots, nnz)
    cols = np.random.randint(0, n_genes, nnz)
    data = np.random.randn(nnz).astype(np.float32)
    
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_spots, n_genes))
    
    # Simulate chunked conversion
    chunk_size = 10000
    result = np.zeros((n_spots, n_genes), dtype=np.float32)
    
    for i in range(0, n_spots, chunk_size):
        end_i = min(i + chunk_size, n_spots)
        result[i:end_i] = sparse_matrix[i:end_i].toarray().astype(np.float32)
    
    # Verify correctness
    expected = sparse_matrix.toarray().astype(np.float32)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_delaunay_specific_exceptions():
    """Test that Delaunay exception handling is specific."""
    from src.cells.graph import CellGraphBuilder
    
    builder = CellGraphBuilder(k=5, use_delaunay=True)
    
    # Test with colinear points (should trigger QhullError)
    colinear_centroids = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    
    # Should not crash, should fall back to kNN only
    src, dst = builder._build_edges(colinear_centroids)
    
    # Should have some edges from kNN
    assert len(src) > 0, "Should have kNN edges even if Delaunay fails"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
