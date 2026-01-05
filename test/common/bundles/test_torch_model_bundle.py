import tempfile
from pathlib import Path
from typing import override

import pytest
import torch
from torch import nn

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.torch_bundle import TorchModelBundle


class SimpleLinearModel(nn.Module):
    """A simple linear model for testing."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(input_size, output_size)
        self.relu: nn.ReLU = nn.ReLU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class ComplexModel(nn.Module):
    """A more complex model with multiple layers for testing."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TestTorchModelBundle:
    """Test suite for TorchModelBundle."""

    def test_simple_model_save_load(self) -> None:
        """Test saving and loading a simple linear model."""
        # Create original model
        original_model = SimpleLinearModel(10, 5)

        # Create some test data and get predictions from original
        test_input = torch.randn(3, 10)
        original_output = original_model(test_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "test_model.bundle"

            # Save model using TorchModelBundle
            model_bundle = TorchModelBundle(original_model)
            metadata = {"test_info": "simple linear model"}
            save_bundle(model_bundle, bundle_path, metadata)

            # Load model back
            loaded_bundle, loaded_metadata = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model

            # Test that loaded model produces same output
            loaded_output = loaded_model(test_input)

            assert torch.allclose(original_output, loaded_output, atol=1e-6)
            assert (
                loaded_metadata["metadata"]["test_info"]
                == "simple linear model"
            )

    def test_complex_model_save_load(self) -> None:
        """Test saving and loading a more complex model."""
        # Create original model
        original_model = ComplexModel(20, 50, 3)

        # Put model in eval mode for consistent behavior
        original_model.eval()

        # Create test data
        test_input = torch.randn(5, 20)
        original_output = original_model(test_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "complex_model.bundle"

            # Save model
            model_bundle = TorchModelBundle(original_model)
            metadata = {"model_type": "complex", "layers": 3}
            save_bundle(model_bundle, bundle_path, metadata)

            # Load model back
            loaded_bundle, loaded_manifest = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model
            loaded_model.eval()  # Ensure eval mode

            # Test that loaded model produces same output
            loaded_output = loaded_model(test_input)

            assert torch.allclose(original_output, loaded_output, atol=1e-6)
            assert loaded_manifest["metadata"]["model_type"] == "complex"

    def test_trained_model_save_load(self) -> None:
        """Test saving and loading a model after training."""
        # Create model and train it briefly
        model = SimpleLinearModel(5, 3)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Generate some training data
        train_input = torch.randn(10, 5)
        train_target = torch.randn(10, 3)

        # Train for a few steps
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            output = model(train_input)
            loss = criterion(output, train_target)
            loss.backward()
            optimizer.step()

        # Get trained model predictions
        model.eval()
        test_input = torch.randn(3, 5)
        trained_output = model(test_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "trained_model.bundle"

            # Save trained model
            model_bundle = TorchModelBundle(model)
            metadata = {"trained": True, "epochs": 10}
            save_bundle(model_bundle, bundle_path, metadata)

            # Load model back
            loaded_bundle, loaded_manifest = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model
            loaded_model.eval()

            # Test that loaded model produces same output
            loaded_output = loaded_model(test_input)

            assert torch.allclose(trained_output, loaded_output, atol=1e-6)
            assert loaded_manifest["metadata"]["trained"] is True

    def test_device_handling(self) -> None:
        """Test device handling functionality."""
        model = SimpleLinearModel(5, 3)
        model_bundle = TorchModelBundle(model)

        # Test initial device (should be CPU)
        assert model_bundle.device == torch.device("cpu")

        # Test moving to different device
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")
            model_bundle.to(cuda_device)
            assert model_bundle.device == cuda_device

            # Move back to CPU
            model_bundle.to(torch.device("cpu"))
            assert model_bundle.device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_save_load(self) -> None:
        """Test saving and loading with CUDA models."""
        # Create model on GPU
        original_model = SimpleLinearModel(8, 4)
        original_model = original_model.cuda()

        # Create test data on GPU
        test_input = torch.randn(2, 8, device="cuda")
        original_output = original_model(test_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "cuda_model.bundle"

            # Save model (should handle device correctly)
            model_bundle = TorchModelBundle(original_model)
            metadata = {"device": "cuda"}
            save_bundle(model_bundle, bundle_path, metadata)

            # Load model back on CPU
            loaded_bundle, _loaded_metadata = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model

            # Test with CPU input
            cpu_input = test_input.cpu()
            loaded_output = loaded_model(cpu_input)
            expected_output = original_output.cpu()

            assert torch.allclose(expected_output, loaded_output, atol=1e-6)

    def test_parameter_preservation(self) -> None:
        """Test that model parameters are preserved correctly."""
        # Create model with specific weights
        model = SimpleLinearModel(3, 2)

        # Set specific weights for testing
        with torch.no_grad():
            model.linear.weight.fill_(0.5)
            model.linear.bias.fill_(0.1)

        # Get original parameters
        original_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "param_test.bundle"

            # Save and load model
            model_bundle = TorchModelBundle(model)
            save_bundle(model_bundle, bundle_path, {"test": "parameters"})

            loaded_bundle, _loaded_metadata = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model

            # Check that parameters match
            for name, param in loaded_model.named_parameters():
                assert torch.equal(original_params[name], param)

    def test_model_state_preservation(self) -> None:
        """Test that model state (training/eval mode) and other attributes are preserved."""
        model = ComplexModel(10, 20, 5)

        # Set to eval mode
        model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "state_test.bundle"

            # Save model
            model_bundle = TorchModelBundle(model)
            save_bundle(model_bundle, bundle_path, {"state": "eval"})

            # Load model
            loaded_bundle, _loaded_metadata = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model

            # Check that eval mode is preserved
            assert not loaded_model.training

            # Test with training mode
            model.train()
            model_bundle = TorchModelBundle(model)
            save_bundle(model_bundle, bundle_path, {"state": "train"})

            loaded_bundle, _loaded_metadata = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )
            assert isinstance(loaded_bundle, TorchModelBundle)
            loaded_model = loaded_bundle.model

            # Check that train mode is preserved
            assert loaded_model.training

    def test_bundle_metadata_integration(self) -> None:
        """Test integration with bundle system metadata."""
        model = SimpleLinearModel(6, 4)
        model_bundle = TorchModelBundle(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "metadata_test.bundle"

            # Save with comprehensive metadata
            metadata = {
                "model_architecture": "SimpleLinearModel",
                "input_size": 6,
                "output_size": 4,
                "created_by": "test_suite",
                "version": "1.0.0",
            }
            save_bundle(model_bundle, bundle_path, metadata)

            # Load and verify manifest
            _loaded_bundle, loaded_manifest = load_bundle(
                bundle_path,
                device=torch.device("cpu"),
            )

            # Check bundle manifest structure
            assert "bundle_version" in loaded_manifest
            assert "class_name" in loaded_manifest
            assert loaded_manifest["class_name"] == "TorchModelBundle"
            assert "metadata" in loaded_manifest

            # Check our custom metadata
            custom_metadata = loaded_manifest["metadata"]
            assert custom_metadata["model_architecture"] == "SimpleLinearModel"
            assert custom_metadata["input_size"] == 6
            assert custom_metadata["output_size"] == 4
            assert custom_metadata["created_by"] == "test_suite"

    def test_class_version(self) -> None:
        """Test that the class version is properly set."""
        model = SimpleLinearModel(2, 2)
        model_bundle = TorchModelBundle(model)

        assert hasattr(model_bundle, "_class_version")
        assert model_bundle._class_version == "1.0.0"  # noqa: SLF001
