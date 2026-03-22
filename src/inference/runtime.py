"""Shared inference runtime selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InferenceBackend = Literal["auto", "torch", "onnx"]
ExecutionProvider = Literal["auto", "cpu", "cuda", "coreml"]

try:
    import torch
except ImportError:  # pragma: no cover - exercised in CPU-only runtime images
    torch = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised in local/dev installs without ORT
    ort = None  # type: ignore[assignment]


TORCH_AVAILABLE = torch is not None
ONNX_AVAILABLE = ort is not None


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved runtime for a model invocation."""

    backend: Literal["torch", "onnx"]
    accelerator: Literal["cpu", "cuda", "mps", "coreml"]
    torch_device: str | None = None
    onnx_provider: str | None = None


def torch_cuda_available() -> bool:
    """Return whether CUDA is available to torch."""
    return bool(TORCH_AVAILABLE and torch.cuda.is_available())


def torch_mps_available() -> bool:
    """Return whether MPS is available to torch."""
    return bool(TORCH_AVAILABLE and torch.backends.mps.is_available())


def gpu_available() -> bool:
    """Expose GPU availability for health/status endpoints."""
    return torch_cuda_available()


def _ensure_torch_available() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "Torch backend requested but 'torch' is not installed. "
            "Install torch or switch the backend to ONNX."
        )


def _ensure_onnx_available() -> None:
    if not ONNX_AVAILABLE:
        raise RuntimeError(
            "ONNX backend requested but 'onnxruntime' is not installed. "
            "Install onnxruntime or switch the backend to torch."
        )


def resolve_runtime(
    backend: InferenceBackend,
    device: str,
    execution_provider: ExecutionProvider,
) -> RuntimeSelection:
    """Resolve the concrete runtime to use for inference."""
    if backend == "torch":
        _ensure_torch_available()
        return _resolve_torch_runtime(device)

    if backend == "onnx":
        _ensure_onnx_available()
        return _resolve_onnx_runtime(device, execution_provider)

    if device == "cuda":
        _ensure_torch_available()
        return _resolve_torch_runtime(device)
    if device == "mps":
        _ensure_torch_available()
        return _resolve_torch_runtime(device)
    if execution_provider != "auto":
        _ensure_onnx_available()
        return _resolve_onnx_runtime(device, execution_provider)

    if torch_cuda_available():
        return RuntimeSelection(
            backend="torch",
            accelerator="cuda",
            torch_device="cuda",
        )
    if torch_mps_available():
        return RuntimeSelection(
            backend="torch",
            accelerator="mps",
            torch_device="mps",
        )
    if ONNX_AVAILABLE:
        return RuntimeSelection(
            backend="onnx",
            accelerator="cpu",
            onnx_provider="CPUExecutionProvider",
        )

    _ensure_torch_available()
    return RuntimeSelection(
        backend="torch",
        accelerator="cpu",
        torch_device="cpu",
    )


def _resolve_torch_runtime(device: str) -> RuntimeSelection:
    _ensure_torch_available()

    if device == "auto":
        if torch_cuda_available():
            return RuntimeSelection(
                backend="torch",
                accelerator="cuda",
                torch_device="cuda",
            )
        if torch_mps_available():
            return RuntimeSelection(
                backend="torch",
                accelerator="mps",
                torch_device="mps",
            )
        return RuntimeSelection(
            backend="torch",
            accelerator="cpu",
            torch_device="cpu",
        )

    if device == "cuda":
        if not torch_cuda_available():
            raise RuntimeError("Torch CUDA backend requested but CUDA is unavailable.")
        return RuntimeSelection(
            backend="torch",
            accelerator="cuda",
            torch_device="cuda",
        )

    if device == "mps":
        if not torch_mps_available():
            raise RuntimeError("Torch MPS backend requested but MPS is unavailable.")
        return RuntimeSelection(
            backend="torch",
            accelerator="mps",
            torch_device="mps",
        )

    return RuntimeSelection(
        backend="torch",
        accelerator="cpu",
        torch_device="cpu",
    )


def _resolve_onnx_runtime(
    device: str,
    execution_provider: ExecutionProvider,
) -> RuntimeSelection:
    _ensure_onnx_available()

    if execution_provider != "auto":
        provider = _provider_name(execution_provider)
    elif device == "cuda":
        provider = "CUDAExecutionProvider"
    elif device == "mps":
        provider = "CoreMLExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    accelerator = _accelerator_name(provider)
    available_providers = set(ort.get_available_providers())
    if provider not in available_providers:
        if provider == "CPUExecutionProvider":
            raise RuntimeError(
                "ONNX CPU provider is unavailable in this environment."
            )
        raise RuntimeError(
            f"Requested ONNX provider '{provider}' is unavailable. "
            f"Available providers: {sorted(available_providers)}"
        )

    return RuntimeSelection(
        backend="onnx",
        accelerator=accelerator,
        onnx_provider=provider,
    )


def _provider_name(execution_provider: ExecutionProvider) -> str:
    return {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }.get(execution_provider, "CPUExecutionProvider")


def _accelerator_name(provider: str) -> Literal["cpu", "cuda", "coreml"]:
    return {
        "CUDAExecutionProvider": "cuda",
        "CoreMLExecutionProvider": "coreml",
    }.get(provider, "cpu")
