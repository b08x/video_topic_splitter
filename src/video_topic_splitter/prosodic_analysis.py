#!/usr/bin/env python3
"""Prosodic analysis functionality using Essentia."""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import essentia.standard as es
import numpy as np

from .constants import CHECKPOINTS
from .project import save_checkpoint

logger = logging.getLogger(__name__)


class ProsodyAnalyzer:
    """Handles extraction and analysis of prosodic features from audio.

    Attributes:
        sample_rate (int): Audio sample rate. Default is 44100 Hz.
        pitch_extractor (es.PitchMelodia): Essentia pitch extractor instance.
        loudness_extractor (es.Loudness): Essentia loudness extractor instance.
        rhythm_extractor (es.RhythmExtractor2013): Essentia rhythm extractor instance.

    """

    # Class-level algorithm instances to prevent "No network created" warning
    pitch_extractor = es.PitchMelodia()
    loudness_extractor = es.Loudness()
    rhythm_extractor = es.RhythmExtractor2013()

    def __init__(self, sample_rate: int = 44100):
        """Initialize the prosody analyzer.

        Args:
            sample_rate: Audio sample rate (default: 44100 Hz)
        """
        self.sample_rate = sample_rate
        self._configure_extractors()

    def _configure_extractors(self):
        """Configures Essentia feature extractors.

        Raises:
            Exception: If there's an error configuring the extractors.

        """
        try:
            # Configure pitch extractor
            self.pitch_extractor.configure(
                sampleRate=self.sample_rate, hopSize=128, frameSize=2048
            )

            # Configure rhythm extractor
            self.rhythm_extractor.configure(method="multifeature")

        except Exception as e:
            logger.error(f"Failed to configure Essentia extractors: {str(e)}")
            raise

    def extract_prosodic_features(self, audio_path: str) -> Dict:
        """Extracts prosodic features from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Dict: A dictionary containing the extracted features: pitch, loudness, and rhythm.
                  Each feature has its values, timestamps, and confidence scores (for pitch and beats).

        Raises:
            Exception: If there's an error during feature extraction.

        """
        try:
            # Load audio
            audio = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)()

            # Extract pitch features
            pitch_values, pitch_confidences = self.pitch_extractor(audio)

            # Extract loudness
            frame_size = 2048
            hop_size = 1024
            frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
            loudness_values = [self.loudness_extractor(frame) for frame in frames]

            # Extract rhythm features
            bpm, beats, beats_confidence, _, beats_intervals = self.rhythm_extractor(
                audio
            )

            features = {
                "pitch": {
                    "values": pitch_values.tolist(),
                    "confidences": pitch_confidences.tolist(),
                    "times": [
                        i * 128 / self.sample_rate for i in range(len(pitch_values))
                    ],
                },
                "loudness": {
                    "values": loudness_values,
                    "times": [
                        i * hop_size / self.sample_rate
                        for i in range(len(loudness_values))
                    ],
                },
                "rhythm": {
                    "bpm": float(bpm),
                    "beats": beats.tolist(),
                    "beats_confidence": float(beats_confidence),
                    "beats_intervals": beats_intervals.tolist(),
                },
            }

            return features

        except Exception as e:
            logger.error(f"Error extracting prosodic features: {str(e)}")
            raise

    def _get_timestamps(self, segment: Dict) -> Tuple[float, float]:
        """Extracts start and end timestamps from a segment.

        Handles different key names ('start_time'/'start' and 'end_time'/'end').

        Args:
            segment (Dict): Transcript segment dictionary.

        Returns:
            Tuple[float, float]: Start and end times as floats.

        Raises:
            ValueError: If timestamps are missing or invalid.

        """
        start = segment.get("start_time")
        if start is None:
            start = segment.get("start")

        end = segment.get("end_time")
        if end is None:
            end = segment.get("end")

        if start is None or end is None:
            raise ValueError(
                f"Transcript segment missing timestamps. Required: 'start_time'/'start' "
                f"and 'end_time'/'end'. Found keys: {list(segment.keys())}"
            )

        try:
            return float(start), float(end)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid timestamp values. Expected floats, got start={start}, end={end}"
            ) from e

    def align_features_with_transcript(
        self, features: Dict, transcript: List[Dict]
    ) -> List[Dict]:
        """Aligns prosodic features with transcript segments.

        Args:
            features (Dict): Extracted prosodic features.
            transcript (List[Dict]): List of transcript segments with timestamps.

        Returns:
            List[Dict]: List of segments with aligned prosodic features.  Each segment will now
                       include a 'prosodic_features' key containing aligned pitch, loudness, and
                       rhythm information.

        Raises:
            Exception: If there's an error during alignment.

        """
        aligned_segments = []

        try:
            for segment in transcript:
                # Get normalized timestamps
                start_time, end_time = self._get_timestamps(segment)

                # Get pitch features within segment timeframe
                pitch_indices = [
                    i
                    for i, t in enumerate(features["pitch"]["times"])
                    if start_time <= t <= end_time
                ]

                segment_pitch = {
                    "values": [features["pitch"]["values"][i] for i in pitch_indices],
                    "confidences": [
                        features["pitch"]["confidences"][i] for i in pitch_indices
                    ],
                    "times": [features["pitch"]["times"][i] for i in pitch_indices],
                }

                # Get loudness features within segment timeframe
                loudness_indices = [
                    i
                    for i, t in enumerate(features["loudness"]["times"])
                    if start_time <= t <= end_time
                ]

                segment_loudness = {
                    "values": [
                        features["loudness"]["values"][i] for i in loudness_indices
                    ],
                    "times": [
                        features["loudness"]["times"][i] for i in loudness_indices
                    ],
                }

                # Get rhythm features within segment timeframe
                segment_beats = [
                    b
                    for b in features["rhythm"]["beats"]
                    if start_time <= b <= end_time
                ]

                segment_rhythm = {
                    "beats": segment_beats,
                    "bpm": features["rhythm"]["bpm"],
                    "beats_confidence": features["rhythm"]["beats_confidence"],
                }

                # Compute summary statistics
                prosodic_features = {
                    "pitch": {
                        "mean": (
                            np.mean(segment_pitch["values"])
                            if segment_pitch["values"]
                            else None
                        ),
                        "std": (
                            np.std(segment_pitch["values"])
                            if segment_pitch["values"]
                            else None
                        ),
                        "range": (
                            (
                                max(segment_pitch["values"])
                                - min(segment_pitch["values"])
                            )
                            if segment_pitch["values"]
                            else None
                        ),
                        "raw": segment_pitch,
                    },
                    "loudness": {
                        "mean": (
                            np.mean(segment_loudness["values"])
                            if segment_loudness["values"]
                            else None
                        ),
                        "std": (
                            np.std(segment_loudness["values"])
                            if segment_loudness["values"]
                            else None
                        ),
                        "range": (
                            (
                                max(segment_loudness["values"])
                                - min(segment_loudness["values"])
                            )
                            if segment_loudness["values"]
                            else None
                        ),
                        "raw": segment_loudness,
                    },
                    "rhythm": segment_rhythm,
                }

                # Add prosodic features to segment
                aligned_segment = {**segment, "prosodic_features": prosodic_features}

                aligned_segments.append(aligned_segment)

            return aligned_segments

        except Exception as e:
            logger.error(f"Error aligning features with transcript: {str(e)}")
            raise

    def process_prosodic_features(
        self, audio_path: str, transcript: List[Dict], project_path: str
    ) -> List[Dict]:
        """Processes audio file to extract and align prosodic features.

        Args:
            audio_path (str): Path to the audio file.
            transcript (List[Dict]): List of transcript segments.
            project_path (str): Path to save results.

        Returns:
            List[Dict]: List of segments with aligned prosodic features.

        Raises:
            Exception: If there's an error during processing.

        """
        try:
            # Extract features
            features = self.extract_prosodic_features(audio_path)

            # Save raw features
            features_path = os.path.join(project_path, "prosodic_features.json")
            with open(features_path, "w") as f:
                json.dump(features, f, indent=2)

            # Save extraction checkpoint
            save_checkpoint(
                project_path,
                CHECKPOINTS["PROSODIC_FEATURES_EXTRACTED"],
                {"features_path": features_path},
            )

            # Align with transcript
            aligned_segments = self.align_features_with_transcript(features, transcript)

            # Save aligned features
            aligned_path = os.path.join(project_path, "aligned_features.json")
            with open(aligned_path, "w") as f:
                json.dump(aligned_segments, f, indent=2)

            # Save alignment checkpoint
            save_checkpoint(
                project_path,
                CHECKPOINTS["PROSODIC_FEATURES_ALIGNED"],
                {"aligned_path": aligned_path},
            )

            return aligned_segments

        except Exception as e:
            logger.error(f"Error processing prosodic features: {str(e)}")
            raise
