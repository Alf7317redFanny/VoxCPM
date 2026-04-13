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
    DEFAULT_LANGUAGE = "zh"  # Chinese by default, per original OpenBMB model

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
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw waveform as a 1-D float32 numpy array, or a path to a
                   WAV/FLAC/MP3 file.
            sample_rate: Sample rate of the provided waveform (ignored when
                         audio is a file path — the file's own rate is used).

        Returns:
            Transcribed text string.
        """
        if not self.is_loaded:
            self.load()

        # Accept file paths as input
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio_file(audio)

        if audio.ndim != 1:
            raise ValueError("Expected a 1-D audio array, got shape %s" % str(audio.shape))

        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=self.language, task="transcribe"
            )
            generated_ids = self._model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
            )

        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return transcription

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio_file(path: Union[str, Path]):
        """Load an audio file and return (waveform_float32, sample_rate)."""
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "soundfile is required to load audio files. "
                "Install it with: pip install soundfile"
            ) from exc

        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
        # Mix down to mono if stereo
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform, sr
