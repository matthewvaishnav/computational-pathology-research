"""Pytest configuration and fixtures for interpretability tests."""

import pytest
import torch
import numpy as np
from hypothesis import settings, Verbosity

# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=50, verbosity=Verbosity.quiet)
settings.register_profile("dev", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile("default")


@pytest.fixture
def device():
    """Provide device for testing (CPU by default, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_cnn_model():
    """Provide a simple mock CNN model for testing."""

    class MockCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.layer2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.layer3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.layer4 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512, 2)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = torch.relu(self.layer4(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return MockCNN()


@pytest.fixture
def sample_image():
    """Provide a sample RGB image for testing."""
    return np.random.rand(96, 96, 3).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Provide a batch of sample images for testing."""
    return torch.randn(4, 3, 96, 96)


@pytest.fixture
def sample_heatmap():
    """Provide a sample heatmap for testing."""
    return np.random.rand(96, 96).astype(np.float32)
