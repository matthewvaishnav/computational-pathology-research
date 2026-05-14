"""
Unit tests for feature-level fusion in TransnnMIL.

Tests the projection layers, cross-attention fusion module, fusion classifier,
and full forward pass integration to ensure correct feature-level fusion behavior.

This test suite validates:
- Projection layers map features to common 512-dimensional space
- Cross-attention fusion combines features from both branches
- Fusion classifier produces correct class predictions
- Full forward pass maintains backward compatibility
- Shape preservation across different batch sizes and configurations
"""

import pytest
import torch

from src.models.transnnmil import TransnnMIL


class TestProjectionLayers:
    """Tests for projection layers (proj_a and proj_b)."""

    def test_proj_a_shape_transformation(self):
        """Test proj_a maps [B, 256] → [B, 512] correctly."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Create dummy Branch A features (CLS token representation)
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            features_a = torch.randn(batch_size, 256)
            projected = model.proj_a(features_a)
            
            assert projected.shape == (batch_size, 512), \
                f"proj_a output shape mismatch for batch_size={batch_size}"
            assert not torch.isnan(projected).any(), \
                f"proj_a produced NaN values for batch_size={batch_size}"

    def test_proj_b_shape_transformation(self):
        """Test proj_b maps [B, 1024] → [B, 512] correctly."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Create dummy Branch B features (aggregated features)
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            features_b = torch.randn(batch_size, 1024)
            projected = model.proj_b(features_b)
            
            assert projected.shape == (batch_size, 512), \
                f"proj_b output shape mismatch for batch_size={batch_size}"
            assert not torch.isnan(projected).any(), \
                f"proj_b produced NaN values for batch_size={batch_size}"

    def test_relu_activation_applied(self):
        """Test ReLU activation is applied (check non-negative outputs)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()  # Set to eval mode to disable dropout
        
        # Create features with both positive and negative values
        features_a = torch.randn(4, 256)
        features_b = torch.randn(4, 1024)
        
        # Project features
        projected_a = model.proj_a(features_a)
        projected_b = model.proj_b(features_b)
        
        # Check that all outputs are non-negative (ReLU applied)
        assert (projected_a >= 0).all(), \
            "proj_a produced negative values, ReLU not applied"
        assert (projected_b >= 0).all(), \
            "proj_b produced negative values, ReLU not applied"

    def test_dropout_applied_during_training(self):
        """Test dropout is applied during training mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.train()  # Set to training mode
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        features_a = torch.randn(8, 256)
        features_b = torch.randn(8, 1024)
        
        # Run multiple forward passes with same input
        torch.manual_seed(42)
        projected_a1 = model.proj_a(features_a)
        projected_b1 = model.proj_b(features_b)
        
        torch.manual_seed(43)  # Different seed
        projected_a2 = model.proj_a(features_a)
        projected_b2 = model.proj_b(features_b)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(projected_a1, projected_a2), \
            "proj_a outputs are identical, dropout not applied"
        assert not torch.allclose(projected_b1, projected_b2), \
            "proj_b outputs are identical, dropout not applied"

    def test_dropout_not_applied_during_eval(self):
        """Test dropout is not applied during eval mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.eval()  # Set to eval mode
        
        features_a = torch.randn(8, 256)
        features_b = torch.randn(8, 1024)
        
        # Run multiple forward passes with same input
        projected_a1 = model.proj_a(features_a)
        projected_b1 = model.proj_b(features_b)
        
        projected_a2 = model.proj_a(features_a)
        projected_b2 = model.proj_b(features_b)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(projected_a1, projected_a2), \
            "proj_a outputs differ in eval mode"
        assert torch.allclose(projected_b1, projected_b2), \
            "proj_b outputs differ in eval mode"

    def test_shape_preservation_across_batch_sizes(self):
        """Test shape preservation across different batch sizes (1, 4, 16, 32)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            # Test proj_a
            features_a = torch.randn(batch_size, 256)
            projected_a = model.proj_a(features_a)
            assert projected_a.shape == (batch_size, 512), \
                f"proj_a shape mismatch for batch_size={batch_size}"
            
            # Test proj_b
            features_b = torch.randn(batch_size, 1024)
            projected_b = model.proj_b(features_b)
            assert projected_b.shape == (batch_size, 512), \
                f"proj_b shape mismatch for batch_size={batch_size}"

    def test_gradient_flow_through_projections(self):
        """Test that gradients flow through projection layers."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        features_a = torch.randn(4, 256, requires_grad=True)
        features_b = torch.randn(4, 1024, requires_grad=True)
        
        projected_a = model.proj_a(features_a)
        projected_b = model.proj_b(features_b)
        
        loss = projected_a.sum() + projected_b.sum()
        loss.backward()
        
        assert features_a.grad is not None, "No gradient for features_a"
        assert features_b.grad is not None, "No gradient for features_b"
        assert features_a.grad.abs().sum() > 0, "Zero gradient for features_a"
        assert features_b.grad.abs().sum() > 0, "Zero gradient for features_b"

    def test_projection_layers_are_sequential(self):
        """Test that projection layers are nn.Sequential modules."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        assert isinstance(model.proj_a, torch.nn.Sequential), \
            "proj_a is not nn.Sequential"
        assert isinstance(model.proj_b, torch.nn.Sequential), \
            "proj_b is not nn.Sequential"
        
        # Check that proj_a has 3 components: Linear, ReLU, Dropout
        assert len(model.proj_a) == 3, \
            f"proj_a should have 3 components, got {len(model.proj_a)}"
        assert isinstance(model.proj_a[0], torch.nn.Linear), \
            "proj_a[0] should be Linear"
        assert isinstance(model.proj_a[1], torch.nn.ReLU), \
            "proj_a[1] should be ReLU"
        assert isinstance(model.proj_a[2], torch.nn.Dropout), \
            "proj_a[2] should be Dropout"
        
        # Check that proj_b has 3 components: Linear, ReLU, Dropout
        assert len(model.proj_b) == 3, \
            f"proj_b should have 3 components, got {len(model.proj_b)}"
        assert isinstance(model.proj_b[0], torch.nn.Linear), \
            "proj_b[0] should be Linear"
        assert isinstance(model.proj_b[1], torch.nn.ReLU), \
            "proj_b[1] should be ReLU"
        assert isinstance(model.proj_b[2], torch.nn.Dropout), \
            "proj_b[2] should be Dropout"

    def test_projection_layer_dimensions(self):
        """Test that projection layers have correct input/output dimensions."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Check proj_a dimensions: 256 → 512
        assert model.proj_a[0].in_features == 256, \
            f"proj_a input dimension should be 256, got {model.proj_a[0].in_features}"
        assert model.proj_a[0].out_features == 512, \
            f"proj_a output dimension should be 512, got {model.proj_a[0].out_features}"
        
        # Check proj_b dimensions: 1024 → 512
        assert model.proj_b[0].in_features == 1024, \
            f"proj_b input dimension should be 1024, got {model.proj_b[0].in_features}"
        assert model.proj_b[0].out_features == 512, \
            f"proj_b output dimension should be 512, got {model.proj_b[0].out_features}"

    def test_projection_dropout_rate(self):
        """Test that projection layers use correct dropout rate."""
        dropout_rate = 0.2
        model = TransnnMIL(
            feature_dim=1024, 
            hidden_dim=256, 
            num_classes=2, 
            dropout=dropout_rate
        )
        
        # Check dropout rate for proj_a
        assert model.proj_a[2].p == dropout_rate, \
            f"proj_a dropout rate should be {dropout_rate}, got {model.proj_a[2].p}"
        
        # Check dropout rate for proj_b
        assert model.proj_b[2].p == dropout_rate, \
            f"proj_b dropout rate should be {dropout_rate}, got {model.proj_b[2].p}"


class TestCrossAttentionFusion:
    """Tests for cross-attention fusion module (fusion_attention).
    
    **Validates: Requirements 10.2**
    """

    def test_fusion_attention_accepts_correct_input_shapes(self):
        """Test fusion_attention accepts correct input shapes [B, 1, 512]."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            # Create query, key, value with shape [B, 1, 512]
            query = torch.randn(batch_size, 1, 512)
            key = torch.randn(batch_size, 1, 512)
            value = torch.randn(batch_size, 1, 512)
            
            # Should not raise any errors
            fused, attn_weights = model.fusion_attention(query, key, value)
            
            assert fused is not None, \
                f"fusion_attention returned None for batch_size={batch_size}"
            assert attn_weights is not None, \
                f"fusion_attention did not return attention weights for batch_size={batch_size}"

    def test_fusion_attention_output_shape_before_squeeze(self):
        """Test output shape is [B, 1, 512] before squeeze."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            # Create query, key, value with shape [B, 1, 512]
            query = torch.randn(batch_size, 1, 512)
            key = torch.randn(batch_size, 1, 512)
            value = torch.randn(batch_size, 1, 512)
            
            # Get output from fusion_attention
            fused, _ = model.fusion_attention(query, key, value)
            
            # Check shape before squeeze
            assert fused.shape == (batch_size, 1, 512), \
                f"fusion_attention output shape mismatch for batch_size={batch_size}: " \
                f"expected ({batch_size}, 1, 512), got {fused.shape}"
            
            # Verify squeeze produces correct shape
            fused_squeezed = fused.squeeze(1)
            assert fused_squeezed.shape == (batch_size, 512), \
                f"Squeezed shape mismatch for batch_size={batch_size}: " \
                f"expected ({batch_size}, 512), got {fused_squeezed.shape}"

    def test_attention_weights_are_computed(self):
        """Test attention weights are computed (second return value)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        query = torch.randn(batch_size, 1, 512)
        key = torch.randn(batch_size, 1, 512)
        value = torch.randn(batch_size, 1, 512)
        
        # Get both outputs
        fused, attn_weights = model.fusion_attention(query, key, value)
        
        # Check that attention weights are returned
        assert attn_weights is not None, \
            "fusion_attention did not return attention weights"
        
        # Check attention weights shape [B, 1, 1] for single query/key
        assert attn_weights.shape == (batch_size, 1, 1), \
            f"Attention weights shape mismatch: expected ({batch_size}, 1, 1), " \
            f"got {attn_weights.shape}"
        
        # Check attention weights are valid probabilities (sum to 1)
        # For single key, attention should be 1.0
        assert torch.allclose(attn_weights, torch.ones_like(attn_weights)), \
            "Attention weights should be 1.0 for single key"
        
        # Check attention weights are non-negative
        assert (attn_weights >= 0).all(), \
            "Attention weights contain negative values"

    def test_multi_head_attention_with_8_heads(self):
        """Test multi-head attention with 8 heads."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Check that fusion_attention has 8 heads
        assert model.fusion_attention.num_heads == 8, \
            f"fusion_attention should have 8 heads, got {model.fusion_attention.num_heads}"
        
        # Check embed_dim is 512
        assert model.fusion_attention.embed_dim == 512, \
            f"fusion_attention embed_dim should be 512, got {model.fusion_attention.embed_dim}"
        
        # Verify head_dim = embed_dim / num_heads = 512 / 8 = 64
        expected_head_dim = 512 // 8
        assert model.fusion_attention.head_dim == expected_head_dim, \
            f"fusion_attention head_dim should be {expected_head_dim}, " \
            f"got {model.fusion_attention.head_dim}"

    def test_batch_first_format_is_used(self):
        """Test batch_first format is used."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Check that batch_first is True
        assert model.fusion_attention.batch_first is True, \
            "fusion_attention should use batch_first=True format"
        
        # Verify by testing with batch_first input format [B, 1, 512]
        batch_size = 4
        query = torch.randn(batch_size, 1, 512)
        key = torch.randn(batch_size, 1, 512)
        value = torch.randn(batch_size, 1, 512)
        
        # Should work without errors with batch_first format
        fused, _ = model.fusion_attention(query, key, value)
        
        # Output should maintain batch_first format [B, 1, 512]
        assert fused.shape[0] == batch_size, \
            f"Output batch dimension mismatch: expected {batch_size}, got {fused.shape[0]}"

    def test_fusion_attention_gradient_flow(self):
        """Test that gradients flow through fusion_attention."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        batch_size = 4
        query = torch.randn(batch_size, 1, 512, requires_grad=True)
        key = torch.randn(batch_size, 1, 512, requires_grad=True)
        value = torch.randn(batch_size, 1, 512, requires_grad=True)
        
        # Forward pass
        fused, _ = model.fusion_attention(query, key, value)
        
        # Backward pass
        loss = fused.sum()
        loss.backward()
        
        # Check gradients exist
        assert query.grad is not None, "No gradient for query"
        assert key.grad is not None, "No gradient for key"
        assert value.grad is not None, "No gradient for value"
        
        # Check gradients are non-zero for value (always has gradient in attention)
        # Note: query gradient can be zero when attention weights are uniform (single key case)
        # key gradient can also be zero in certain attention configurations
        assert value.grad.abs().sum() > 0, "Zero gradient for value"

    def test_fusion_attention_dropout_applied_during_training(self):
        """Test dropout is applied during training mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.train()  # Set to training mode
        
        batch_size = 8
        query = torch.randn(batch_size, 1, 512)
        key = torch.randn(batch_size, 1, 512)
        value = torch.randn(batch_size, 1, 512)
        
        # Run multiple forward passes with same input
        torch.manual_seed(42)
        fused1, _ = model.fusion_attention(query, key, value)
        
        torch.manual_seed(43)  # Different seed
        fused2, _ = model.fusion_attention(query, key, value)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(fused1, fused2), \
            "fusion_attention outputs are identical, dropout not applied"

    def test_fusion_attention_no_dropout_during_eval(self):
        """Test dropout is not applied during eval mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.eval()  # Set to eval mode
        
        batch_size = 8
        query = torch.randn(batch_size, 1, 512)
        key = torch.randn(batch_size, 1, 512)
        value = torch.randn(batch_size, 1, 512)
        
        # Run multiple forward passes with same input
        fused1, _ = model.fusion_attention(query, key, value)
        fused2, _ = model.fusion_attention(query, key, value)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(fused1, fused2), \
            "fusion_attention outputs differ in eval mode"

    def test_fusion_attention_with_different_query_key_values(self):
        """Test fusion_attention with different query and key/value features."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        
        # Create different features for query (Branch A) and key/value (Branch B)
        query = torch.randn(batch_size, 1, 512)
        key = torch.randn(batch_size, 1, 512)
        value = torch.randn(batch_size, 1, 512)
        
        # Forward pass
        fused, attn_weights = model.fusion_attention(query, key, value)
        
        # Check output shape
        assert fused.shape == (batch_size, 1, 512), \
            f"Output shape mismatch: expected ({batch_size}, 1, 512), got {fused.shape}"
        
        # Check attention weights shape
        assert attn_weights.shape == (batch_size, 1, 1), \
            f"Attention weights shape mismatch: expected ({batch_size}, 1, 1), " \
            f"got {attn_weights.shape}"

    def test_fusion_attention_numerical_stability(self):
        """Test fusion_attention handles numerical stability (no NaN/Inf)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        
        # Test with normal values
        query = torch.randn(batch_size, 1, 512)
        key = torch.randn(batch_size, 1, 512)
        value = torch.randn(batch_size, 1, 512)
        
        fused, attn_weights = model.fusion_attention(query, key, value)
        
        # Check for NaN or Inf
        assert not torch.isnan(fused).any(), \
            "fusion_attention produced NaN values"
        assert not torch.isinf(fused).any(), \
            "fusion_attention produced Inf values"
        assert not torch.isnan(attn_weights).any(), \
            "Attention weights contain NaN values"
        assert not torch.isinf(attn_weights).any(), \
            "Attention weights contain Inf values"

    def test_fusion_attention_is_multihead_attention(self):
        """Test that fusion_attention is nn.MultiheadAttention module."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        assert isinstance(model.fusion_attention, torch.nn.MultiheadAttention), \
            "fusion_attention is not nn.MultiheadAttention"

    def test_fusion_attention_dropout_rate(self):
        """Test that fusion_attention uses correct dropout rate."""
        dropout_rate = 0.2
        model = TransnnMIL(
            feature_dim=1024, 
            hidden_dim=256, 
            num_classes=2, 
            dropout=dropout_rate
        )
        
        # Check dropout rate
        assert model.fusion_attention.dropout == dropout_rate, \
            f"fusion_attention dropout rate should be {dropout_rate}, " \
            f"got {model.fusion_attention.dropout}"


class TestFusionClassifier:
    """Tests for fusion classifier.
    
    **Validates: Requirements 10.3**
    """

    def test_fusion_classifier_shape_transformation(self):
        """Test 512 → 256 → num_classes mapping."""
        # Test with different num_classes values
        num_classes_list = [2, 3, 5]
        
        for num_classes in num_classes_list:
            model = TransnnMIL(
                feature_dim=1024, 
                hidden_dim=256, 
                num_classes=num_classes
            )
            model.eval()
            
            # Test with different batch sizes
            batch_sizes = [1, 4, 16, 32]
            
            for batch_size in batch_sizes:
                # Create dummy fused features [B, 512]
                fused_features = torch.randn(batch_size, 512)
                
                # Pass through fusion classifier
                logits = model.fusion_classifier(fused_features)
                
                # Check output shape [B, num_classes]
                assert logits.shape == (batch_size, num_classes), \
                    f"fusion_classifier output shape mismatch for " \
                    f"batch_size={batch_size}, num_classes={num_classes}: " \
                    f"expected ({batch_size}, {num_classes}), got {logits.shape}"
                
                # Check no NaN values
                assert not torch.isnan(logits).any(), \
                    f"fusion_classifier produced NaN values for " \
                    f"batch_size={batch_size}, num_classes={num_classes}"

    def test_relu_activation_between_layers(self):
        """Test ReLU activation between layers (check non-negative intermediate outputs)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()  # Set to eval mode to disable dropout
        
        batch_size = 4
        fused_features = torch.randn(batch_size, 512)
        
        # Get intermediate output after first layer (512 → 256) and ReLU
        # fusion_classifier structure: Linear(512, 256), ReLU, Dropout, Linear(256, num_classes)
        intermediate = model.fusion_classifier[0](fused_features)  # Linear layer
        intermediate = model.fusion_classifier[1](intermediate)     # ReLU activation
        
        # Check that all intermediate outputs are non-negative (ReLU applied)
        assert (intermediate >= 0).all(), \
            "fusion_classifier intermediate outputs contain negative values, ReLU not applied"
        
        # Verify intermediate shape is [B, 256]
        assert intermediate.shape == (batch_size, 256), \
            f"Intermediate shape mismatch: expected ({batch_size}, 256), got {intermediate.shape}"

    def test_dropout_applied_during_training(self):
        """Test dropout is applied during training mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.train()  # Set to training mode
        
        batch_size = 8
        fused_features = torch.randn(batch_size, 512)
        
        # Run multiple forward passes with same input
        torch.manual_seed(42)
        logits1 = model.fusion_classifier(fused_features)
        
        torch.manual_seed(43)  # Different seed
        logits2 = model.fusion_classifier(fused_features)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(logits1, logits2), \
            "fusion_classifier outputs are identical, dropout not applied"

    def test_dropout_not_applied_during_eval(self):
        """Test dropout is not applied during eval mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.eval()  # Set to eval mode
        
        batch_size = 8
        fused_features = torch.randn(batch_size, 512)
        
        # Run multiple forward passes with same input
        logits1 = model.fusion_classifier(fused_features)
        logits2 = model.fusion_classifier(fused_features)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(logits1, logits2), \
            "fusion_classifier outputs differ in eval mode"

    def test_output_shape_matches_num_classes(self):
        """Test output shape matches [B, num_classes] for different num_classes (2, 3, 5)."""
        num_classes_list = [2, 3, 5]
        batch_sizes = [1, 4, 16, 32]
        
        for num_classes in num_classes_list:
            model = TransnnMIL(
                feature_dim=1024, 
                hidden_dim=256, 
                num_classes=num_classes
            )
            model.eval()
            
            for batch_size in batch_sizes:
                # Create dummy fused features
                fused_features = torch.randn(batch_size, 512)
                
                # Get logits
                logits = model.fusion_classifier(fused_features)
                
                # Verify shape
                assert logits.shape == (batch_size, num_classes), \
                    f"Output shape mismatch for batch_size={batch_size}, " \
                    f"num_classes={num_classes}: expected ({batch_size}, {num_classes}), " \
                    f"got {logits.shape}"

    def test_fusion_classifier_is_sequential(self):
        """Test that fusion_classifier is nn.Sequential module."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        assert isinstance(model.fusion_classifier, torch.nn.Sequential), \
            "fusion_classifier is not nn.Sequential"
        
        # Check that fusion_classifier has 4 components: Linear, ReLU, Dropout, Linear
        assert len(model.fusion_classifier) == 4, \
            f"fusion_classifier should have 4 components, got {len(model.fusion_classifier)}"
        
        assert isinstance(model.fusion_classifier[0], torch.nn.Linear), \
            "fusion_classifier[0] should be Linear"
        assert isinstance(model.fusion_classifier[1], torch.nn.ReLU), \
            "fusion_classifier[1] should be ReLU"
        assert isinstance(model.fusion_classifier[2], torch.nn.Dropout), \
            "fusion_classifier[2] should be Dropout"
        assert isinstance(model.fusion_classifier[3], torch.nn.Linear), \
            "fusion_classifier[3] should be Linear"

    def test_fusion_classifier_layer_dimensions(self):
        """Test that fusion_classifier layers have correct dimensions."""
        num_classes_list = [2, 3, 5]
        
        for num_classes in num_classes_list:
            model = TransnnMIL(
                feature_dim=1024, 
                hidden_dim=256, 
                num_classes=num_classes
            )
            
            # Check first layer: 512 → 256
            assert model.fusion_classifier[0].in_features == 512, \
                f"First layer input dimension should be 512, " \
                f"got {model.fusion_classifier[0].in_features}"
            assert model.fusion_classifier[0].out_features == 256, \
                f"First layer output dimension should be 256, " \
                f"got {model.fusion_classifier[0].out_features}"
            
            # Check second layer: 256 → num_classes
            assert model.fusion_classifier[3].in_features == 256, \
                f"Second layer input dimension should be 256, " \
                f"got {model.fusion_classifier[3].in_features}"
            assert model.fusion_classifier[3].out_features == num_classes, \
                f"Second layer output dimension should be {num_classes}, " \
                f"got {model.fusion_classifier[3].out_features}"

    def test_fusion_classifier_dropout_rate(self):
        """Test that fusion_classifier uses correct dropout rate."""
        dropout_rate = 0.2
        model = TransnnMIL(
            feature_dim=1024, 
            hidden_dim=256, 
            num_classes=2, 
            dropout=dropout_rate
        )
        
        # Check dropout rate
        assert model.fusion_classifier[2].p == dropout_rate, \
            f"fusion_classifier dropout rate should be {dropout_rate}, " \
            f"got {model.fusion_classifier[2].p}"

    def test_fusion_classifier_gradient_flow(self):
        """Test that gradients flow through fusion_classifier."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        batch_size = 4
        fused_features = torch.randn(batch_size, 512, requires_grad=True)
        
        # Forward pass
        logits = model.fusion_classifier(fused_features)
        
        # Backward pass
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist
        assert fused_features.grad is not None, \
            "No gradient for fused_features"
        
        # Check gradients are non-zero
        assert fused_features.grad.abs().sum() > 0, \
            "Zero gradient for fused_features"

    def test_fusion_classifier_numerical_stability(self):
        """Test fusion_classifier handles numerical stability (no NaN/Inf)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        
        # Test with normal values
        fused_features = torch.randn(batch_size, 512)
        logits = model.fusion_classifier(fused_features)
        
        # Check for NaN or Inf
        assert not torch.isnan(logits).any(), \
            "fusion_classifier produced NaN values"
        assert not torch.isinf(logits).any(), \
            "fusion_classifier produced Inf values"

    def test_fusion_classifier_with_different_batch_sizes(self):
        """Test fusion_classifier works with different batch sizes."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            fused_features = torch.randn(batch_size, 512)
            logits = model.fusion_classifier(fused_features)
            
            assert logits.shape == (batch_size, 2), \
                f"Output shape mismatch for batch_size={batch_size}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"
            
            assert not torch.isnan(logits).any(), \
                f"NaN values for batch_size={batch_size}"

    def test_fusion_classifier_output_range(self):
        """Test that fusion_classifier outputs logits (unbounded values)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 100
        fused_features = torch.randn(batch_size, 512)
        
        logits = model.fusion_classifier(fused_features)
        
        # Logits should be unbounded (not probabilities)
        # Check that we have values outside [0, 1] range
        assert (logits < 0).any() or (logits > 1).any(), \
            "fusion_classifier outputs appear to be bounded to [0, 1], " \
            "should output logits (unbounded)"


class TestFullForwardPassIntegration:
    """Integration tests for full forward pass with feature-level fusion.
    
    **Validates: Requirements 10.4, 6.2, 6.3, 6.4, 7.3**
    """

    def test_end_to_end_forward_pass_with_feature_fusion(self):
        """Test end-to-end forward pass with feature-level fusion."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        # Create dummy input: [batch_size, num_patches, feature_dim]
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        num_patches_tensor = torch.tensor([100, 80, 90, 100])
        
        # Forward pass
        logits = model(features, num_patches_tensor, return_attention=False)
        
        # Check output shape
        assert logits.shape == (batch_size, 2), \
            f"Output shape mismatch: expected ({batch_size}, 2), got {logits.shape}"
        
        # Check no NaN values
        assert not torch.isnan(logits).any(), \
            "Forward pass produced NaN values"
        
        # Check logits are unbounded (not probabilities)
        assert (logits < 0).any() or (logits > 1).any(), \
            "Logits appear to be bounded to [0, 1], should be unbounded"

    def test_return_attention_true_returns_tuple_with_correct_shapes(self):
        """Test return_attention=True returns tuple (logits, attention_weights) with correct shapes."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        num_patches_tensor = torch.tensor([100, 80, 90, 100])
        
        # Forward pass with return_attention=True
        result = model(features, num_patches_tensor, return_attention=True)
        
        # Check that result is a tuple
        assert isinstance(result, tuple), \
            f"Expected tuple, got {type(result)}"
        
        # Check tuple has 2 elements
        assert len(result) == 2, \
            f"Expected tuple of length 2, got {len(result)}"
        
        logits, attention_weights = result
        
        # Check logits shape
        assert logits.shape == (batch_size, 2), \
            f"Logits shape mismatch: expected ({batch_size}, 2), got {logits.shape}"
        
        # Check attention weights shape (from Branch A)
        assert attention_weights.shape == (batch_size, num_patches), \
            f"Attention weights shape mismatch: expected ({batch_size}, {num_patches}), " \
            f"got {attention_weights.shape}"
        
        # Check attention weights are valid probabilities
        # Sum should be close to 1 for each sample
        attention_sums = attention_weights.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            f"Attention weights do not sum to 1: {attention_sums}"
        
        # Check attention weights are non-negative
        assert (attention_weights >= 0).all(), \
            "Attention weights contain negative values"

    def test_return_attention_false_returns_only_logits_tensor(self):
        """Test return_attention=False returns only logits tensor."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        num_patches_tensor = torch.tensor([100, 80, 90, 100])
        
        # Forward pass with return_attention=False
        result = model(features, num_patches_tensor, return_attention=False)
        
        # Check that result is a tensor (not a tuple)
        assert isinstance(result, torch.Tensor), \
            f"Expected torch.Tensor, got {type(result)}"
        
        # Check shape
        assert result.shape == (batch_size, 2), \
            f"Output shape mismatch: expected ({batch_size}, 2), got {result.shape}"

    def test_forward_pass_with_variable_length_bags(self):
        """Test with variable-length bags using different num_patches values."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        max_patches = 150
        features = torch.randn(batch_size, max_patches, 1024)
        
        # Different actual patch counts per sample
        num_patches_tensor = torch.tensor([150, 100, 80, 120])
        
        # Forward pass
        logits = model(features, num_patches_tensor, return_attention=False)
        
        # Check output shape
        assert logits.shape == (batch_size, 2), \
            f"Output shape mismatch: expected ({batch_size}, 2), got {logits.shape}"
        
        # Check no NaN values
        assert not torch.isnan(logits).any(), \
            "Forward pass with variable-length bags produced NaN values"
        
        # Test with return_attention=True
        logits, attention = model(features, num_patches_tensor, return_attention=True)
        
        # Check shapes
        assert logits.shape == (batch_size, 2), \
            f"Logits shape mismatch: expected ({batch_size}, 2), got {logits.shape}"
        assert attention.shape == (batch_size, max_patches), \
            f"Attention shape mismatch: expected ({batch_size}, {max_patches}), " \
            f"got {attention.shape}"

    def test_forward_pass_with_fixed_length_bags_no_num_patches(self):
        """Test with fixed-length bags (num_patches=None)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Forward pass without num_patches (all bags have same length)
        logits = model(features, num_patches=None, return_attention=False)
        
        # Check output shape
        assert logits.shape == (batch_size, 2), \
            f"Output shape mismatch: expected ({batch_size}, 2), got {logits.shape}"
        
        # Check no NaN values
        assert not torch.isnan(logits).any(), \
            "Forward pass with fixed-length bags produced NaN values"
        
        # Test with return_attention=True
        logits, attention = model(features, num_patches=None, return_attention=True)
        
        # Check shapes
        assert logits.shape == (batch_size, 2), \
            f"Logits shape mismatch: expected ({batch_size}, 2), got {logits.shape}"
        assert attention.shape == (batch_size, num_patches), \
            f"Attention shape mismatch: expected ({batch_size}, {num_patches}), " \
            f"got {attention.shape}"

    def test_output_logits_have_correct_shape_for_different_num_classes(self):
        """Test output logits have shape [batch_size, num_classes]."""
        num_classes_list = [2, 3, 5, 10]
        batch_sizes = [1, 4, 16]
        
        for num_classes in num_classes_list:
            model = TransnnMIL(
                feature_dim=1024, 
                hidden_dim=256, 
                num_classes=num_classes
            )
            model.eval()
            
            for batch_size in batch_sizes:
                num_patches = 100
                features = torch.randn(batch_size, num_patches, 1024)
                
                # Forward pass
                logits = model(features, num_patches=None, return_attention=False)
                
                # Check output shape
                assert logits.shape == (batch_size, num_classes), \
                    f"Output shape mismatch for batch_size={batch_size}, " \
                    f"num_classes={num_classes}: expected ({batch_size}, {num_classes}), " \
                    f"got {logits.shape}"

    def test_forward_pass_with_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        num_patches = 100
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, num_patches, 1024)
            
            # Forward pass
            logits = model(features, num_patches=None, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, 2), \
                f"Output shape mismatch for batch_size={batch_size}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"
            
            # Check no NaN values
            assert not torch.isnan(logits).any(), \
                f"Forward pass produced NaN values for batch_size={batch_size}"

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through the entire forward pass."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024, requires_grad=True)
        
        # Forward pass
        logits = model(features, num_patches=None, return_attention=False)
        
        # Backward pass
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist
        assert features.grad is not None, \
            "No gradient for input features"
        
        # Check gradients are non-zero
        assert features.grad.abs().sum() > 0, \
            "Zero gradient for input features"
        
        # Check gradients for fusion components
        assert model.proj_a[0].weight.grad is not None, \
            "No gradient for proj_a"
        assert model.proj_b[0].weight.grad is not None, \
            "No gradient for proj_b"
        assert model.fusion_classifier[0].weight.grad is not None, \
            "No gradient for fusion_classifier"

    def test_forward_pass_numerical_stability(self):
        """Test forward pass handles numerical stability (no NaN/Inf)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        
        # Test with normal values
        features = torch.randn(batch_size, num_patches, 1024)
        logits = model(features, num_patches=None, return_attention=False)
        
        assert not torch.isnan(logits).any(), \
            "Forward pass produced NaN values"
        assert not torch.isinf(logits).any(), \
            "Forward pass produced Inf values"
        
        # Test with large values
        features_large = torch.randn(batch_size, num_patches, 1024) * 10
        logits_large = model(features_large, num_patches=None, return_attention=False)
        
        assert not torch.isnan(logits_large).any(), \
            "Forward pass produced NaN values with large inputs"
        assert not torch.isinf(logits_large).any(), \
            "Forward pass produced Inf values with large inputs"

    def test_forward_pass_deterministic_in_eval_mode(self):
        """Test forward pass is deterministic in eval mode."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Run forward pass twice with same input
        logits1 = model(features, num_patches=None, return_attention=False)
        logits2 = model(features, num_patches=None, return_attention=False)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(logits1, logits2), \
            "Forward pass outputs differ in eval mode (should be deterministic)"

    def test_forward_pass_stochastic_in_train_mode(self):
        """Test forward pass is stochastic in train mode (due to dropout)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5)
        model.train()  # Set to training mode
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Run forward pass multiple times with same input
        torch.manual_seed(42)
        logits1 = model(features, num_patches=None, return_attention=False)
        
        torch.manual_seed(43)  # Different seed
        logits2 = model(features, num_patches=None, return_attention=False)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(logits1, logits2), \
            "Forward pass outputs are identical in train mode (dropout not applied)"

    def test_forward_pass_with_single_sample(self):
        """Test forward pass with batch_size=1."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        # Single sample
        features = torch.randn(1, 100, 1024)
        
        # Forward pass
        logits = model(features, num_patches=None, return_attention=False)
        
        # Check output shape
        assert logits.shape == (1, 2), \
            f"Output shape mismatch: expected (1, 2), got {logits.shape}"
        
        # Test with return_attention=True
        logits, attention = model(features, num_patches=None, return_attention=True)
        
        assert logits.shape == (1, 2), \
            f"Logits shape mismatch: expected (1, 2), got {logits.shape}"
        assert attention.shape == (1, 100), \
            f"Attention shape mismatch: expected (1, 100), got {attention.shape}"

    def test_forward_pass_with_different_feature_dimensions(self):
        """Test forward pass with different input feature dimensions."""
        feature_dims = [512, 768, 1024, 2048]
        
        for feature_dim in feature_dims:
            model = TransnnMIL(
                feature_dim=feature_dim, 
                hidden_dim=256, 
                num_classes=2
            )
            model.eval()
            
            batch_size = 4
            num_patches = 100
            features = torch.randn(batch_size, num_patches, feature_dim)
            
            # Forward pass
            logits = model(features, num_patches=None, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, 2), \
                f"Output shape mismatch for feature_dim={feature_dim}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"

    def test_forward_pass_with_different_num_patches(self):
        """Test forward pass with different numbers of patches."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches_list = [10, 50, 100, 200, 500]
        
        for num_patches in num_patches_list:
            features = torch.randn(batch_size, num_patches, 1024)
            
            # Forward pass
            logits = model(features, num_patches=None, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, 2), \
                f"Output shape mismatch for num_patches={num_patches}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"

    def test_forward_pass_integration_with_all_components(self):
        """Test that forward pass correctly integrates all fusion components."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Forward pass
        logits = model(features, num_patches=None, return_attention=False)
        
        # Manually trace through the pipeline to verify integration
        with torch.no_grad():
            # Extract features from both branches
            features_a = model.branch_a.get_features(features, num_patches=None)
            features_b = model.branch_b.get_features(features, num_patches=None)
            
            # Project to common dimension
            proj_a = model.proj_a(features_a)
            proj_b = model.proj_b(features_b)
            
            # Reshape for attention
            query = proj_a.unsqueeze(1)
            key = proj_b.unsqueeze(1)
            value = proj_b.unsqueeze(1)
            
            # Apply fusion attention
            fused, _ = model.fusion_attention(query, key, value)
            fused = fused.squeeze(1)
            
            # Apply fusion classifier
            expected_logits = model.fusion_classifier(fused)
        
        # Check that forward pass produces same result as manual pipeline
        assert torch.allclose(logits, expected_logits, atol=1e-5), \
            "Forward pass output does not match manual pipeline execution"



class TestGetBranchOutputs:
    """Tests for get_branch_outputs() functionality.
    
    **Validates: Requirements 10.5, 5.1, 5.2, 5.3, 5.4**
    """

    def test_returns_three_outputs(self):
        """Test get_branch_outputs returns three outputs (logits_a, logits_b, logits_fused)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Call get_branch_outputs
        result = model.get_branch_outputs(features)
        
        # Check that it returns a tuple of 3 elements
        assert isinstance(result, tuple), \
            f"get_branch_outputs should return a tuple, got {type(result)}"
        assert len(result) == 3, \
            f"get_branch_outputs should return 3 outputs, got {len(result)}"
        
        logits_a, logits_b, logits_fused = result
        
        # Check that all outputs are tensors
        assert isinstance(logits_a, torch.Tensor), \
            f"logits_a should be a tensor, got {type(logits_a)}"
        assert isinstance(logits_b, torch.Tensor), \
            f"logits_b should be a tensor, got {type(logits_b)}"
        assert isinstance(logits_fused, torch.Tensor), \
            f"logits_fused should be a tensor, got {type(logits_fused)}"

    def test_all_outputs_have_correct_shape(self):
        """Test all outputs have correct shape [batch_size, num_classes]."""
        # Test with different batch sizes and num_classes
        batch_sizes = [1, 4, 16, 32]
        num_classes_list = [2, 3, 5]
        
        for batch_size in batch_sizes:
            for num_classes in num_classes_list:
                model = TransnnMIL(
                    feature_dim=1024, 
                    hidden_dim=256, 
                    num_classes=num_classes
                )
                model.eval()
                
                num_patches = 100
                features = torch.randn(batch_size, num_patches, 1024)
                
                # Get branch outputs
                logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
                
                # Check shapes
                expected_shape = (batch_size, num_classes)
                
                assert logits_a.shape == expected_shape, \
                    f"logits_a shape mismatch for batch_size={batch_size}, " \
                    f"num_classes={num_classes}: expected {expected_shape}, " \
                    f"got {logits_a.shape}"
                
                assert logits_b.shape == expected_shape, \
                    f"logits_b shape mismatch for batch_size={batch_size}, " \
                    f"num_classes={num_classes}: expected {expected_shape}, " \
                    f"got {logits_b.shape}"
                
                assert logits_fused.shape == expected_shape, \
                    f"logits_fused shape mismatch for batch_size={batch_size}, " \
                    f"num_classes={num_classes}: expected {expected_shape}, " \
                    f"got {logits_fused.shape}"

    def test_outputs_are_different(self):
        """Test outputs are different (fusion has measurable effect)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 16
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Get branch outputs
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Check that logits_a and logits_b are different
        # (they come from different branches)
        assert not torch.allclose(logits_a, logits_b, rtol=1e-3, atol=1e-3), \
            "logits_a and logits_b are too similar, branches may not be different"
        
        # Check that logits_fused is different from both branches
        # (fusion should produce different results)
        assert not torch.allclose(logits_fused, logits_a, rtol=1e-3, atol=1e-3), \
            "logits_fused and logits_a are too similar, fusion may not have effect"
        
        assert not torch.allclose(logits_fused, logits_b, rtol=1e-3, atol=1e-3), \
            "logits_fused and logits_b are too similar, fusion may not have effect"

    def test_no_grad_context_is_used(self):
        """Test no_grad context is used (verify no gradients are computed)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.train()  # Set to training mode to enable gradient computation
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024, requires_grad=True)
        
        # Call get_branch_outputs
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Check that outputs do not require gradients
        # (because get_branch_outputs uses torch.no_grad())
        assert not logits_a.requires_grad, \
            "logits_a requires gradients, no_grad context not used"
        assert not logits_b.requires_grad, \
            "logits_b requires gradients, no_grad context not used"
        assert not logits_fused.requires_grad, \
            "logits_fused requires gradients, no_grad context not used"
        
        # Verify that we cannot compute gradients from these outputs
        try:
            loss = logits_a.sum() + logits_b.sum() + logits_fused.sum()
            loss.backward()
            # If we reach here, gradients were computed (should not happen)
            assert False, \
                "Gradients were computed from get_branch_outputs, no_grad not working"
        except RuntimeError as e:
            # Expected: RuntimeError because tensors don't require gradients
            assert "does not require grad" in str(e).lower() or \
                   "element 0 of tensors does not require grad" in str(e).lower(), \
                f"Unexpected error: {e}"

    def test_works_with_variable_length_bags(self):
        """Test get_branch_outputs works with variable-length bags (num_patches parameter)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        max_patches = 200
        features = torch.randn(batch_size, max_patches, 1024)
        
        # Create variable patch counts
        num_patches = torch.tensor([50, 100, 150, 200])
        
        # Call get_branch_outputs with num_patches
        logits_a, logits_b, logits_fused = model.get_branch_outputs(
            features, num_patches=num_patches
        )
        
        # Check shapes
        expected_shape = (batch_size, 2)
        assert logits_a.shape == expected_shape, \
            f"logits_a shape mismatch with num_patches: expected {expected_shape}, " \
            f"got {logits_a.shape}"
        assert logits_b.shape == expected_shape, \
            f"logits_b shape mismatch with num_patches: expected {expected_shape}, " \
            f"got {logits_b.shape}"
        assert logits_fused.shape == expected_shape, \
            f"logits_fused shape mismatch with num_patches: expected {expected_shape}, " \
            f"got {logits_fused.shape}"
        
        # Check no NaN values
        assert not torch.isnan(logits_a).any(), \
            "logits_a contains NaN with variable-length bags"
        assert not torch.isnan(logits_b).any(), \
            "logits_b contains NaN with variable-length bags"
        assert not torch.isnan(logits_fused).any(), \
            "logits_fused contains NaN with variable-length bags"

    def test_works_without_num_patches(self):
        """Test get_branch_outputs works without num_patches (fixed-length bags)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Call get_branch_outputs without num_patches
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Check shapes
        expected_shape = (batch_size, 2)
        assert logits_a.shape == expected_shape, \
            f"logits_a shape mismatch: expected {expected_shape}, got {logits_a.shape}"
        assert logits_b.shape == expected_shape, \
            f"logits_b shape mismatch: expected {expected_shape}, got {logits_b.shape}"
        assert logits_fused.shape == expected_shape, \
            f"logits_fused shape mismatch: expected {expected_shape}, got {logits_fused.shape}"

    def test_numerical_stability(self):
        """Test get_branch_outputs handles numerical stability (no NaN/Inf)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Get branch outputs
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Check for NaN
        assert not torch.isnan(logits_a).any(), \
            "logits_a contains NaN values"
        assert not torch.isnan(logits_b).any(), \
            "logits_b contains NaN values"
        assert not torch.isnan(logits_fused).any(), \
            "logits_fused contains NaN values"
        
        # Check for Inf
        assert not torch.isinf(logits_a).any(), \
            "logits_a contains Inf values"
        assert not torch.isinf(logits_b).any(), \
            "logits_b contains Inf values"
        assert not torch.isinf(logits_fused).any(), \
            "logits_fused contains Inf values"

    def test_consistency_with_forward_pass(self):
        """Test that logits_fused from get_branch_outputs matches forward() output."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Get fused logits from get_branch_outputs
        _, _, logits_fused_from_method = model.get_branch_outputs(features)
        
        # Get logits from forward pass
        logits_from_forward = model(features)
        
        # They should be identical (both use same fusion pipeline)
        assert torch.allclose(logits_fused_from_method, logits_from_forward, rtol=1e-5, atol=1e-5), \
            "logits_fused from get_branch_outputs does not match forward() output"

    def test_branch_outputs_use_original_classifiers(self):
        """Test that logits_a and logits_b use original branch classifiers."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Get branch outputs
        logits_a, logits_b, _ = model.get_branch_outputs(features)
        
        # Get logits directly from branches
        logits_a_direct = model.branch_a(features, return_attention=False)
        logits_b_direct = model.branch_b(features, return_attention=False)
        
        # They should be identical
        assert torch.allclose(logits_a, logits_a_direct, rtol=1e-5, atol=1e-5), \
            "logits_a from get_branch_outputs does not match direct branch_a output"
        
        assert torch.allclose(logits_b, logits_b_direct, rtol=1e-5, atol=1e-5), \
            "logits_b from get_branch_outputs does not match direct branch_b output"

    def test_efficiency_single_forward_pass(self):
        """Test that get_branch_outputs computes all three outputs efficiently."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # This test verifies that get_branch_outputs runs without errors
        # and produces valid outputs in a single call
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Verify all outputs are valid
        assert logits_a is not None and logits_a.numel() > 0, \
            "logits_a is empty or None"
        assert logits_b is not None and logits_b.numel() > 0, \
            "logits_b is empty or None"
        assert logits_fused is not None and logits_fused.numel() > 0, \
            "logits_fused is empty or None"
        
        # Verify outputs have expected properties
        assert logits_a.dtype == torch.float32, \
            f"logits_a has wrong dtype: {logits_a.dtype}"
        assert logits_b.dtype == torch.float32, \
            f"logits_b has wrong dtype: {logits_b.dtype}"
        assert logits_fused.dtype == torch.float32, \
            f"logits_fused has wrong dtype: {logits_fused.dtype}"

    def test_device_compatibility(self):
        """Test get_branch_outputs works on different devices (CPU)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        
        # Test on CPU
        features_cpu = torch.randn(batch_size, num_patches, 1024)
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features_cpu)
        
        # Check that outputs are on CPU
        assert logits_a.device.type == 'cpu', \
            f"logits_a should be on CPU, got {logits_a.device}"
        assert logits_b.device.type == 'cpu', \
            f"logits_b should be on CPU, got {logits_b.device}"
        assert logits_fused.device.type == 'cpu', \
            f"logits_fused should be on CPU, got {logits_fused.device}"
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            features_cuda = features_cpu.cuda()
            
            logits_a_cuda, logits_b_cuda, logits_fused_cuda = \
                model_cuda.get_branch_outputs(features_cuda)
            
            # Check that outputs are on CUDA
            assert logits_a_cuda.device.type == 'cuda', \
                f"logits_a should be on CUDA, got {logits_a_cuda.device}"
            assert logits_b_cuda.device.type == 'cuda', \
                f"logits_b should be on CUDA, got {logits_b_cuda.device}"
            assert logits_fused_cuda.device.type == 'cuda', \
                f"logits_fused should be on CUDA, got {logits_fused_cuda.device}"

    def test_batch_size_one(self):
        """Test get_branch_outputs works with batch_size=1."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 1
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Get branch outputs
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Check shapes
        expected_shape = (1, 2)
        assert logits_a.shape == expected_shape, \
            f"logits_a shape mismatch: expected {expected_shape}, got {logits_a.shape}"
        assert logits_b.shape == expected_shape, \
            f"logits_b shape mismatch: expected {expected_shape}, got {logits_b.shape}"
        assert logits_fused.shape == expected_shape, \
            f"logits_fused shape mismatch: expected {expected_shape}, got {logits_fused.shape}"

    def test_large_batch_size(self):
        """Test get_branch_outputs works with large batch sizes."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 64
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Get branch outputs
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        
        # Check shapes
        expected_shape = (batch_size, 2)
        assert logits_a.shape == expected_shape, \
            f"logits_a shape mismatch: expected {expected_shape}, got {logits_a.shape}"
        assert logits_b.shape == expected_shape, \
            f"logits_b shape mismatch: expected {expected_shape}, got {logits_b.shape}"
        assert logits_fused.shape == expected_shape, \
            f"logits_fused shape mismatch: expected {expected_shape}, got {logits_fused.shape}"
        
        # Check no NaN values
        assert not torch.isnan(logits_a).any(), \
            "logits_a contains NaN with large batch size"
        assert not torch.isnan(logits_b).any(), \
            "logits_b contains NaN with large batch size"
        assert not torch.isnan(logits_fused).any(), \
            "logits_fused contains NaN with large batch size"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code.
    
    **Validates: Requirements 10.6, 6.1, 6.2, 6.5, 7.1**
    """

    def test_constructor_accepts_same_parameters(self):
        """Test constructor accepts same parameters as before."""
        # Test with all standard parameters
        model = TransnnMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            num_layers=2,
            num_heads=8
        )
        
        # Verify model was created successfully
        assert model is not None, "Model creation failed"
        assert isinstance(model, TransnnMIL), "Model is not TransnnMIL instance"
        
        # Test with minimal parameters
        model_minimal = TransnnMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=3
        )
        
        assert model_minimal is not None, "Minimal model creation failed"
        
        # Test with different feature dimensions
        for feature_dim in [512, 768, 1024, 2048]:
            model_var = TransnnMIL(
                feature_dim=feature_dim,
                hidden_dim=256,
                num_classes=2
            )
            assert model_var is not None, \
                f"Model creation failed for feature_dim={feature_dim}"

    def test_forward_signature_unchanged(self):
        """Test forward() signature unchanged (features, num_patches, return_attention)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        num_patches_tensor = torch.tensor([100, 80, 90, 100])
        
        # Test with all parameters
        logits = model(features, num_patches_tensor, return_attention=False)
        assert logits.shape == (batch_size, 2), \
            "forward() with all parameters failed"
        
        # Test with positional arguments
        logits_pos = model(features, num_patches_tensor, False)
        assert logits_pos.shape == (batch_size, 2), \
            "forward() with positional arguments failed"
        
        # Test with return_attention=True
        result = model(features, num_patches_tensor, return_attention=True)
        assert isinstance(result, tuple), \
            "forward() with return_attention=True should return tuple"
        assert len(result) == 2, \
            "forward() should return (logits, attention)"
        
        # Test without num_patches
        logits_no_np = model(features, return_attention=False)
        assert logits_no_np.shape == (batch_size, 2), \
            "forward() without num_patches failed"
        
        # Test with only features (default parameters)
        logits_default = model(features)
        assert logits_default.shape == (batch_size, 2), \
            "forward() with only features failed"

    def test_get_gate_value_method_exists_and_works(self):
        """Test get_gate_value() method still exists and works."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Check method exists
        assert hasattr(model, 'get_gate_value'), \
            "get_gate_value() method does not exist"
        
        # Check method is callable
        assert callable(model.get_gate_value), \
            "get_gate_value is not callable"
        
        # Call method and verify it returns a value
        gate_value = model.get_gate_value()
        
        # Check return type
        assert isinstance(gate_value, (int, float, torch.Tensor)), \
            f"get_gate_value() should return numeric value, got {type(gate_value)}"
        
        # If tensor, check it's a scalar
        if isinstance(gate_value, torch.Tensor):
            assert gate_value.numel() == 1, \
                "get_gate_value() should return scalar tensor"
            gate_value = gate_value.item()
        
        # Check value is in valid range [0, 1] (sigmoid output)
        assert 0 <= gate_value <= 1, \
            f"get_gate_value() should return value in [0, 1], got {gate_value}"

    def test_gate_param_exists_as_nn_parameter(self):
        """Test gate_param exists as nn.Parameter."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Check gate_param exists
        assert hasattr(model, 'gate_param'), \
            "gate_param attribute does not exist"
        
        # Check it's an nn.Parameter
        assert isinstance(model.gate_param, torch.nn.Parameter), \
            f"gate_param should be nn.Parameter, got {type(model.gate_param)}"
        
        # Check it requires grad
        assert model.gate_param.requires_grad, \
            "gate_param should require gradients"
        
        # Check shape (should be scalar)
        assert model.gate_param.numel() == 1, \
            f"gate_param should be scalar, got shape {model.gate_param.shape}"
        
        # Check it's in model parameters
        param_names = [name for name, _ in model.named_parameters()]
        assert 'gate_param' in param_names, \
            "gate_param not in model.named_parameters()"

    def test_model_works_with_nnmil_trainer_interface(self):
        """Test model works with nnMILTrainer interface (mock trainer test)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        # Simulate nnMILTrainer usage patterns
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Pattern 1: Standard forward pass
        logits = model(features)
        assert logits.shape == (batch_size, 2), \
            "Standard forward pass failed"
        
        # Pattern 2: Get branch outputs for loss computation
        logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
        assert logits_a.shape == (batch_size, 2), \
            "Branch A output shape incorrect"
        assert logits_b.shape == (batch_size, 2), \
            "Branch B output shape incorrect"
        assert logits_fused.shape == (batch_size, 2), \
            "Fused output shape incorrect"
        
        # Pattern 3: Get gate value for logging
        gate_value = model.get_gate_value()
        assert isinstance(gate_value, (int, float, torch.Tensor)), \
            "get_gate_value() failed"
        
        # Pattern 4: Training mode
        model.train()
        logits_train = model(features)
        assert logits_train.shape == (batch_size, 2), \
            "Training mode forward pass failed"
        
        # Pattern 5: Gradient computation
        model.zero_grad()
        loss = logits_train.sum()
        loss.backward()
        
        # Check gradients exist for fusion components (not gate_param)
        # Note: gate_param is maintained for backward compatibility but not used
        # in feature-level fusion, so it doesn't receive gradients
        assert model.proj_a[0].weight.grad is not None, \
            "proj_a gradient not computed"
        assert model.proj_b[0].weight.grad is not None, \
            "proj_b gradient not computed"
        assert model.fusion_classifier[0].weight.grad is not None, \
            "fusion_classifier gradient not computed"
        
        # Pattern 6: Eval mode
        model.eval()
        with torch.no_grad():
            logits_eval = model(features)
        assert logits_eval.shape == (batch_size, 2), \
            "Eval mode forward pass failed"

    def test_model_state_dict_compatible(self):
        """Test model state_dict is compatible (can save/load)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Check it's a dict
        assert isinstance(state_dict, dict), \
            "state_dict() should return dict"
        
        # Check key components exist in state dict
        key_components = [
            'gate_param',
            'proj_a.0.weight',  # proj_a Linear layer
            'proj_b.0.weight',  # proj_b Linear layer
            'fusion_classifier.0.weight',  # fusion_classifier first Linear
            'fusion_classifier.3.weight',  # fusion_classifier second Linear
        ]
        
        for key in key_components:
            assert key in state_dict, \
                f"Key component '{key}' not in state_dict"
        
        # Test save and load
        # Create new model and load state
        model_new = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model_new.load_state_dict(state_dict)
        
        # Verify loaded model works
        model_new.eval()
        features = torch.randn(4, 100, 1024)
        logits = model_new(features)
        assert logits.shape == (4, 2), \
            "Loaded model forward pass failed"

    def test_model_to_device_works(self):
        """Test model.to(device) works correctly."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Test CPU
        model_cpu = model.to('cpu')
        assert model_cpu is not None, \
            "model.to('cpu') failed"
        
        # Verify forward pass works on CPU
        features_cpu = torch.randn(4, 100, 1024)
        logits_cpu = model_cpu(features_cpu)
        assert logits_cpu.device.type == 'cpu', \
            "Output not on CPU"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            features_cuda = torch.randn(4, 100, 1024, device='cuda')
            logits_cuda = model_cuda(features_cuda)
            assert logits_cuda.device.type == 'cuda', \
                "Output not on CUDA"

    def test_model_train_eval_modes_work(self):
        """Test model.train() and model.eval() modes work correctly."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Test train mode
        model.train()
        assert model.training, \
            "model.train() did not set training=True"
        
        # Test eval mode
        model.eval()
        assert not model.training, \
            "model.eval() did not set training=False"
        
        # Test forward pass in both modes
        features = torch.randn(4, 100, 1024)
        
        model.train()
        logits_train = model(features)
        assert logits_train.shape == (4, 2), \
            "Forward pass in train mode failed"
        
        model.eval()
        logits_eval = model(features)
        assert logits_eval.shape == (4, 2), \
            "Forward pass in eval mode failed"

    def test_model_parameters_accessible(self):
        """Test model.parameters() returns all parameters."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Get parameters
        params = list(model.parameters())
        
        # Check we have parameters
        assert len(params) > 0, \
            "model.parameters() returned empty list"
        
        # Check all are tensors
        for param in params:
            assert isinstance(param, torch.nn.Parameter), \
                "model.parameters() contains non-Parameter"
        
        # Check named_parameters works
        named_params = dict(model.named_parameters())
        
        # Check key parameters exist
        assert 'gate_param' in named_params, \
            "gate_param not in named_parameters"
        assert any('proj_a' in name for name in named_params), \
            "proj_a parameters not in named_parameters"
        assert any('proj_b' in name for name in named_params), \
            "proj_b parameters not in named_parameters"
        assert any('fusion_classifier' in name for name in named_params), \
            "fusion_classifier parameters not in named_parameters"

    def test_model_repr_works(self):
        """Test model.__repr__() works (for debugging/logging)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Get string representation
        repr_str = repr(model)
        
        # Check it's a string
        assert isinstance(repr_str, str), \
            "repr() should return string"
        
        # Check it's not empty
        assert len(repr_str) > 0, \
            "repr() returned empty string"
        
        # Check it contains model name
        assert 'TransnnMIL' in repr_str, \
            "repr() should contain model name"



class TestShapeValidation:
    """Tests for shape validation across the model.
    
    **Validates: Requirements 10.7**
    """

    def test_all_intermediate_shapes_correct(self):
        """Test all intermediate shapes are correct (features_a, features_b, proj_a, proj_b, fused, logits)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        with torch.no_grad():
            # Extract features from both branches
            features_a = model.branch_a.get_features(features, num_patches=None)
            features_b = model.branch_b.get_features(features, num_patches=None)
            
            # Check branch feature shapes
            assert features_a.shape == (batch_size, 256), \
                f"features_a shape incorrect: expected ({batch_size}, 256), got {features_a.shape}"
            assert features_b.shape == (batch_size, 1024), \
                f"features_b shape incorrect: expected ({batch_size}, 1024), got {features_b.shape}"
            
            # Project to common dimension
            proj_a = model.proj_a(features_a)
            proj_b = model.proj_b(features_b)
            
            # Check projection shapes
            assert proj_a.shape == (batch_size, 512), \
                f"proj_a shape incorrect: expected ({batch_size}, 512), got {proj_a.shape}"
            assert proj_b.shape == (batch_size, 512), \
                f"proj_b shape incorrect: expected ({batch_size}, 512), got {proj_b.shape}"
            
            # Reshape for attention
            query = proj_a.unsqueeze(1)
            key = proj_b.unsqueeze(1)
            value = proj_b.unsqueeze(1)
            
            # Check reshaped shapes
            assert query.shape == (batch_size, 1, 512), \
                f"query shape incorrect: expected ({batch_size}, 1, 512), got {query.shape}"
            assert key.shape == (batch_size, 1, 512), \
                f"key shape incorrect: expected ({batch_size}, 1, 512), got {key.shape}"
            assert value.shape == (batch_size, 1, 512), \
                f"value shape incorrect: expected ({batch_size}, 1, 512), got {value.shape}"
            
            # Apply fusion attention
            fused, _ = model.fusion_attention(query, key, value)
            
            # Check fused shape before squeeze
            assert fused.shape == (batch_size, 1, 512), \
                f"fused shape before squeeze incorrect: expected ({batch_size}, 1, 512), got {fused.shape}"
            
            # Squeeze
            fused = fused.squeeze(1)
            
            # Check fused shape after squeeze
            assert fused.shape == (batch_size, 512), \
                f"fused shape after squeeze incorrect: expected ({batch_size}, 512), got {fused.shape}"
            
            # Apply fusion classifier
            logits = model.fusion_classifier(fused)
            
            # Check final logits shape
            assert logits.shape == (batch_size, 2), \
                f"logits shape incorrect: expected ({batch_size}, 2), got {logits.shape}"

    def test_error_handling_for_wrong_feature_dim(self):
        """Test error handling for mismatched input feature dimension."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        
        # Create features with wrong dimension (512 instead of 1024)
        features_wrong = torch.randn(batch_size, num_patches, 512)
        
        # Should raise error due to dimension mismatch
        with pytest.raises(RuntimeError):
            model(features_wrong, num_patches=None, return_attention=False)

    def test_error_handling_for_wrong_num_patches(self):
        """Test error handling for mismatched num_patches."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        
        # Create num_patches tensor with wrong batch size
        num_patches_wrong = torch.tensor([100, 80, 90])  # Only 3 elements, need 4
        
        # Should raise error due to batch size mismatch
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            model(features, num_patches=num_patches_wrong, return_attention=False)

    def test_batch_size_variations(self):
        """Test batch size variations (1, 4, 16, 32)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        num_patches = 100
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, num_patches, 1024)
            
            # Forward pass
            logits = model(features, num_patches=None, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, 2), \
                f"Output shape incorrect for batch_size={batch_size}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"
            
            # Check no NaN
            assert not torch.isnan(logits).any(), \
                f"NaN values for batch_size={batch_size}"

    def test_num_patches_variations(self):
        """Test num_patches variations (different values per batch)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        max_patches = 200
        
        # Test different num_patches configurations
        num_patches_configs = [
            torch.tensor([50, 100, 150, 200]),
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([100, 100, 100, 100]),
            torch.tensor([200, 200, 200, 200]),
        ]
        
        for num_patches in num_patches_configs:
            features = torch.randn(batch_size, max_patches, 1024)
            
            # Forward pass
            logits = model(features, num_patches=num_patches, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, 2), \
                f"Output shape incorrect for num_patches={num_patches.tolist()}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"
            
            # Check no NaN
            assert not torch.isnan(logits).any(), \
                f"NaN values for num_patches={num_patches.tolist()}"

    def test_3d_input_validation(self):
        """Test 3D input validation (reject 2D or 4D inputs)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        # Test 2D input (should fail)
        features_2d = torch.randn(4, 1024)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            model(features_2d, num_patches=None, return_attention=False)
        
        # Test 4D input (should fail)
        features_4d = torch.randn(4, 100, 1024, 1)
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            model(features_4d, num_patches=None, return_attention=False)
        
        # Test 3D input (should work)
        features_3d = torch.randn(4, 100, 1024)
        logits = model(features_3d, num_patches=None, return_attention=False)
        assert logits.shape == (4, 2), \
            "3D input should work"

    def test_shape_preservation_with_different_num_classes(self):
        """Test shape preservation with different num_classes."""
        num_classes_list = [2, 3, 5, 10, 20]
        batch_size = 4
        num_patches = 100
        
        for num_classes in num_classes_list:
            model = TransnnMIL(
                feature_dim=1024, 
                hidden_dim=256, 
                num_classes=num_classes
            )
            model.eval()
            
            features = torch.randn(batch_size, num_patches, 1024)
            
            # Forward pass
            logits = model(features, num_patches=None, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, num_classes), \
                f"Output shape incorrect for num_classes={num_classes}: " \
                f"expected ({batch_size}, {num_classes}), got {logits.shape}"

    def test_shape_preservation_with_different_feature_dims(self):
        """Test shape preservation with different feature_dim."""
        feature_dims = [512, 768, 1024, 2048]
        batch_size = 4
        num_patches = 100
        
        for feature_dim in feature_dims:
            model = TransnnMIL(
                feature_dim=feature_dim, 
                hidden_dim=256, 
                num_classes=2
            )
            model.eval()
            
            features = torch.randn(batch_size, num_patches, feature_dim)
            
            # Forward pass
            logits = model(features, num_patches=None, return_attention=False)
            
            # Check output shape
            assert logits.shape == (batch_size, 2), \
                f"Output shape incorrect for feature_dim={feature_dim}: " \
                f"expected ({batch_size}, 2), got {logits.shape}"

    def test_attention_weights_shape_matches_num_patches(self):
        """Test attention weights shape matches num_patches."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches_list = [50, 100, 200, 500]
        
        for num_patches in num_patches_list:
            features = torch.randn(batch_size, num_patches, 1024)
            
            # Forward pass with return_attention=True
            logits, attention = model(features, num_patches=None, return_attention=True)
            
            # Check attention shape matches num_patches
            assert attention.shape == (batch_size, num_patches), \
                f"Attention shape incorrect for num_patches={num_patches}: " \
                f"expected ({batch_size}, {num_patches}), got {attention.shape}"

    def test_get_branch_outputs_shape_consistency(self):
        """Test get_branch_outputs returns consistent shapes."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        num_patches = 100
        
        for batch_size in batch_sizes:
            features = torch.randn(batch_size, num_patches, 1024)
            
            # Get branch outputs
            logits_a, logits_b, logits_fused = model.get_branch_outputs(features)
            
            # Check all have same shape
            expected_shape = (batch_size, 2)
            assert logits_a.shape == expected_shape, \
                f"logits_a shape incorrect for batch_size={batch_size}"
            assert logits_b.shape == expected_shape, \
                f"logits_b shape incorrect for batch_size={batch_size}"
            assert logits_fused.shape == expected_shape, \
                f"logits_fused shape incorrect for batch_size={batch_size}"


class TestDeviceCompatibility:
    """Tests for device compatibility (CPU and CUDA).
    
    **Validates: Requirements 7.2**
    """

    def test_model_works_on_cpu(self):
        """Test model works on CPU."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model = model.to('cpu')
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024, device='cpu')
        
        # Forward pass
        logits = model(features, num_patches=None, return_attention=False)
        
        # Check output is on CPU
        assert logits.device.type == 'cpu', \
            f"Output should be on CPU, got {logits.device}"
        
        # Check output shape
        assert logits.shape == (batch_size, 2), \
            f"Output shape incorrect: expected ({batch_size}, 2), got {logits.shape}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_works_on_cuda(self):
        """Test model works on CUDA (if available, skip if not)."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model = model.to('cuda')
        model.eval()
        
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024, device='cuda')
        
        # Forward pass
        logits = model(features, num_patches=None, return_attention=False)
        
        # Check output is on CUDA
        assert logits.device.type == 'cuda', \
            f"Output should be on CUDA, got {logits.device}"
        
        # Check output shape
        assert logits.shape == (batch_size, 2), \
            f"Output shape incorrect: expected ({batch_size}, 2), got {logits.shape}"

    def test_model_handles_device_transfers(self):
        """Test model handles device transfers correctly (.to(device))."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Start on CPU
        model = model.to('cpu')
        
        # Check all components are on CPU
        assert model.gate_param.device.type == 'cpu', \
            "gate_param not on CPU"
        assert model.proj_a[0].weight.device.type == 'cpu', \
            "proj_a not on CPU"
        assert model.proj_b[0].weight.device.type == 'cpu', \
            "proj_b not on CPU"
        assert model.fusion_classifier[0].weight.device.type == 'cpu', \
            "fusion_classifier not on CPU"
        
        # Transfer to CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            
            # Check all components moved to CUDA
            assert model.gate_param.device.type == 'cuda', \
                "gate_param not on CUDA"
            assert model.proj_a[0].weight.device.type == 'cuda', \
                "proj_a not on CUDA"
            assert model.proj_b[0].weight.device.type == 'cuda', \
                "proj_b not on CUDA"
            assert model.fusion_classifier[0].weight.device.type == 'cuda', \
                "fusion_classifier not on CUDA"
            
            # Transfer back to CPU
            model = model.to('cpu')
            
            # Check all components moved back to CPU
            assert model.gate_param.device.type == 'cpu', \
                "gate_param not back on CPU"
            assert model.proj_a[0].weight.device.type == 'cpu', \
                "proj_a not back on CPU"
            assert model.proj_b[0].weight.device.type == 'cpu', \
                "proj_b not back on CPU"
            assert model.fusion_classifier[0].weight.device.type == 'cpu', \
                "fusion_classifier not back on CPU"

    def test_all_components_move_to_same_device(self):
        """Test all components move to same device."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        # Move to CPU
        model = model.to('cpu')
        
        # Get all parameter devices
        devices = set()
        for param in model.parameters():
            devices.add(param.device.type)
        
        # All should be on same device (CPU)
        assert len(devices) == 1, \
            f"Parameters on multiple devices: {devices}"
        assert 'cpu' in devices, \
            "Not all parameters on CPU"
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            
            # Get all parameter devices
            devices = set()
            for param in model.parameters():
                devices.add(param.device.type)
            
            # All should be on same device (CUDA)
            assert len(devices) == 1, \
                f"Parameters on multiple devices: {devices}"
            assert 'cuda' in devices, \
                "Not all parameters on CUDA"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass_produces_same_results_on_cpu_and_cuda(self):
        """Test forward pass produces same results on CPU and CUDA (within tolerance)."""
        # Create model
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        features_cpu = torch.randn(batch_size, num_patches, 1024)
        
        # CPU forward pass
        model_cpu = model.to('cpu')
        with torch.no_grad():
            logits_cpu = model_cpu(features_cpu, num_patches=None, return_attention=False)
        
        # CUDA forward pass
        model_cuda = model.to('cuda')
        features_cuda = features_cpu.to('cuda')
        with torch.no_grad():
            logits_cuda = model_cuda(features_cuda, num_patches=None, return_attention=False)
        
        # Move CUDA results back to CPU for comparison
        logits_cuda_cpu = logits_cuda.cpu()
        
        # Check results are close (within tolerance)
        assert torch.allclose(logits_cpu, logits_cuda_cpu, rtol=1e-4, atol=1e-5), \
            "CPU and CUDA results differ beyond tolerance"

    def test_get_branch_outputs_device_consistency(self):
        """Test get_branch_outputs maintains device consistency."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        
        # Test on CPU
        model_cpu = model.to('cpu')
        features_cpu = torch.randn(batch_size, num_patches, 1024, device='cpu')
        
        logits_a, logits_b, logits_fused = model_cpu.get_branch_outputs(features_cpu)
        
        assert logits_a.device.type == 'cpu', \
            "logits_a not on CPU"
        assert logits_b.device.type == 'cpu', \
            "logits_b not on CPU"
        assert logits_fused.device.type == 'cpu', \
            "logits_fused not on CPU"
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            features_cuda = torch.randn(batch_size, num_patches, 1024, device='cuda')
            
            logits_a, logits_b, logits_fused = model_cuda.get_branch_outputs(features_cuda)
            
            assert logits_a.device.type == 'cuda', \
                "logits_a not on CUDA"
            assert logits_b.device.type == 'cuda', \
                "logits_b not on CUDA"
            assert logits_fused.device.type == 'cuda', \
                "logits_fused not on CUDA"

    def test_attention_weights_device_consistency(self):
        """Test attention weights maintain device consistency."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        model.eval()
        
        batch_size = 4
        num_patches = 100
        
        # Test on CPU
        model_cpu = model.to('cpu')
        features_cpu = torch.randn(batch_size, num_patches, 1024, device='cpu')
        
        logits, attention = model_cpu(features_cpu, num_patches=None, return_attention=True)
        
        assert logits.device.type == 'cpu', \
            "logits not on CPU"
        assert attention.device.type == 'cpu', \
            "attention not on CPU"
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            features_cuda = torch.randn(batch_size, num_patches, 1024, device='cuda')
            
            logits, attention = model_cuda(features_cuda, num_patches=None, return_attention=True)
            
            assert logits.device.type == 'cuda', \
                "logits not on CUDA"
            assert attention.device.type == 'cuda', \
                "attention not on CUDA"

    def test_gradient_computation_on_different_devices(self):
        """Test gradient computation works on different devices."""
        model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        
        batch_size = 4
        num_patches = 100
        
        # Test on CPU
        model_cpu = model.to('cpu')
        features_cpu = torch.randn(batch_size, num_patches, 1024, device='cpu', requires_grad=True)
        
        logits_cpu = model_cpu(features_cpu, num_patches=None, return_attention=False)
        loss_cpu = logits_cpu.sum()
        loss_cpu.backward()
        
        assert features_cpu.grad is not None, \
            "No gradient on CPU"
        assert features_cpu.grad.device.type == 'cpu', \
            "Gradient not on CPU"
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            features_cuda = torch.randn(batch_size, num_patches, 1024, device='cuda', requires_grad=True)
            
            logits_cuda = model_cuda(features_cuda, num_patches=None, return_attention=False)
            loss_cuda = logits_cuda.sum()
            loss_cuda.backward()
            
            assert features_cuda.grad is not None, \
                "No gradient on CUDA"
            assert features_cuda.grad.device.type == 'cuda', \
                "Gradient not on CUDA"
