"""Core VoxCPM model wrapper for speech recognition."""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class VoxCPMModel:
    """Wrapper around the VoxCPM speech recognition model.

    Handles model loading, audio preprocessing, and inference.
    """

    DEFAULT_SAMPLE_RATE = 16000
    # Changed default to English since I'm primarily using this for English audio
    DEFAULT_LANGUAGE = "en"

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
        language: str = DEFAULT_LANGUAGE,
    ):
        """
        Args:
            model_dir: Path to the directory containing model weights and config.
            device: Torch device string (e.g. 'cpu', 'cuda:0'). Auto-detected if None.
            language: Target language for recognition (e.g. 'zh', 'en').
        """
        self.model_dir = Path(model_dir)
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

        logger.info("VoxCPMModel initialised (device=%s, language=%s)", self.device, self.language)

    def load(self) -> None:
        """Load model weights and processor from model_dir."""
        if self._model is not None:
            logger.debug("Model already loaded, skipping.")
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        logger.info("Loading VoxCPM model from %s ...", self.model_dir)

        try:
            # Lazy import so the package stays importable without heavy deps
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            self._processor = AutoProcessor.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)
            self._model.eval()
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded into memory."""
        return self._model is not None

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        # NOTE: setting a reasonable max length to avoid runaway generation on long files
        max_new_tokens: int = 256,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw waveform as a 1-D float32 numpy array, or a path to a
                   WAV/FLAC/MP3 file.
            sample_rate: Sample rate of the provided waveform (ignored when
                         audio is a file path — the file's own rate is used).
            max_new_tokens: Maximum number of tokens to generate. Increase for
                            very long audio segments.

        Returns:
            Transcribed text str
