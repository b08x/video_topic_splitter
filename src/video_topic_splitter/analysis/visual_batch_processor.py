#!/usr/bin/env python3
"""
Advanced batch processing for cross-segment visual analysis optimization.

This module provides sophisticated batching strategies for processing multiple video
segments' frames collectively, maximizing the efficiency of Gemini API batch processing
by consolidating requests across segment boundaries.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import hashlib

from PIL import Image
from ..api.gemini import batch_analyze_images_with_gemini
from ..progress_tracker import ProgressTracker
from .batch_config import get_batch_config, optimize_batch_size_for_workload

logger = logging.getLogger(__name__)


@dataclass
class FrameRequest:
    """Data class for individual frame analysis requests."""
    frame_path: str
    segment_id: str
    frame_number: int
    timestamp: float
    prompt: str
    metadata: Dict[str, Any]
    
    def to_batch_request(self) -> Dict[str, Any]:
        """Convert to batch API request format."""
        return {
            'prompt': self.prompt,
            'image': self.frame_path,
            'frame_id': f"{self.segment_id}_frame_{self.frame_number}",
            'metadata': {
                'segment_id': self.segment_id,
                'frame_number': self.frame_number,
                'timestamp': self.timestamp,
                **self.metadata
            }
        }


class VisualBatchProcessor:
    """
    Advanced batch processor for optimizing visual analysis across multiple segments.
    
    This class implements intelligent batching strategies that collect frames from
    multiple video segments and process them together using Gemini's batch API,
    achieving maximum efficiency and cost savings.
    """
    
    def __init__(self, progress_tracker: ProgressTracker = None, batch_size: int = None):
        """
        Initialize the batch processor.
        
        Args:
            progress_tracker: Optional progress tracker for monitoring
            batch_size: Maximum number of frames per batch (auto-optimized if None)
        """
        self.progress_tracker = progress_tracker
        self.config = get_batch_config()
        self.batch_size = batch_size or self.config.optimal_batch_size
        self.pending_requests: List[FrameRequest] = []
        self.completed_analyses: Dict[str, Dict[str, Any]] = {}
        self.processing_stats = {
            'total_frames_processed': 0,
            'total_batches_submitted': 0,
            'batch_processing_time': 0,
            'average_batch_size': 0
        }
    
    def add_frame_request(
        self, 
        frame_path: str, 
        segment_id: str, 
        frame_number: int, 
        timestamp: float,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a frame analysis request to the pending batch.
        
        Args:
            frame_path: Path to the frame image file
            segment_id: Unique identifier for the segment
            frame_number: Frame number within the segment
            timestamp: Timestamp of the frame in the video
            prompt: Analysis prompt for Gemini
            metadata: Additional metadata for the frame
        """
        request = FrameRequest(
            frame_path=frame_path,
            segment_id=segment_id,
            frame_number=frame_number,
            timestamp=timestamp,
            prompt=prompt,
            metadata=metadata or {}
        )
        
        self.pending_requests.append(request)
        
        logger.debug(f"Added frame request: {segment_id}_frame_{frame_number}")
    
    def should_flush_batch(self) -> bool:
        """
        Determine if the current batch should be processed.
        
        Returns:
            True if batch should be processed now, False otherwise
        """
        return len(self.pending_requests) >= self.batch_size
    
    def process_pending_batch(self, force_process: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all pending frame requests as a batch.
        
        Args:
            force_process: Process even if batch is smaller than optimal size
            
        Returns:
            Dictionary mapping segment_id to list of frame analysis results
        """
        if not self.pending_requests:
            return {}
        
        if not force_process and not self.should_flush_batch():
            logger.debug(f"Batch size ({len(self.pending_requests)}) below threshold, waiting for more requests")
            return {}
        
        logger.info(f"Processing batch of {len(self.pending_requests)} frames across segments")
        
        start_time = time.time()
        
        # Convert to batch API format
        batch_requests = [req.to_batch_request() for req in self.pending_requests]
        
        # Create progress callback
        def batch_progress_callback(progress: float, message: str):
            if self.progress_tracker:
                self.progress_tracker.update_phase_progress(
                    progress, f"Batch Processing: {message}"
                )
        
        # Process the batch
        try:
            batch_results = batch_analyze_images_with_gemini(
                batch_requests,
                progress_callback=batch_progress_callback
            )
            
            # Organize results by segment
            segment_results = defaultdict(list)
            
            for result in batch_results:
                metadata = result.get('metadata', {})
                segment_id = metadata.get('segment_id', 'unknown')
                
                # Add processing metrics to metadata
                result_metadata = {
                    **metadata,
                    'batch_processing_timestamp': time.time(),
                    'batch_size': len(batch_requests)
                }
                
                segment_result = {
                    'frame_number': metadata.get('frame_number', 0),
                    'timestamp': metadata.get('timestamp', 0),
                    'frame_path': next(
                        req.frame_path for req in self.pending_requests 
                        if req.segment_id == segment_id and req.frame_number == metadata.get('frame_number', 0)
                    ),
                    'metadata': result_metadata
                }
                
                if 'error' in result:
                    segment_result['error'] = result['error']
                else:
                    segment_result['analysis'] = result.get('analysis', '')
                
                segment_results[segment_id].append(segment_result)
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self.processing_stats['total_frames_processed'] += len(self.pending_requests)
            self.processing_stats['total_batches_submitted'] += 1
            self.processing_stats['batch_processing_time'] += processing_time
            
            if self.processing_stats['total_batches_submitted'] > 0:
                self.processing_stats['average_batch_size'] = (
                    self.processing_stats['total_frames_processed'] / 
                    self.processing_stats['total_batches_submitted']
                )
            
            # Clear pending requests
            processed_count = len(self.pending_requests)
            self.pending_requests = []
            
            logger.info(
                f"Batch processing completed: {processed_count} frames processed in {processing_time:.2f}s "
                f"across {len(segment_results)} segments"
            )
            
            return dict(segment_results)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Clear pending requests to prevent reprocessing
            self.pending_requests = []
            raise
    
    def get_segment_results(self, segment_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve analysis results for a specific segment.
        
        Args:
            segment_id: The segment identifier
            
        Returns:
            List of frame analysis results for the segment
        """
        return self.completed_analyses.get(segment_id, [])
    
    def finalize_processing(self) -> Dict[str, Any]:
        """
        Process any remaining pending requests and return final statistics.
        
        Returns:
            Dictionary containing processing statistics and performance metrics
        """
        # Process any remaining pending requests
        if self.pending_requests:
            logger.info(f"Processing final batch of {len(self.pending_requests)} remaining frames")
            final_results = self.process_pending_batch(force_process=True)
            self.completed_analyses.update(final_results)
        
        # Calculate performance metrics
        total_time = self.processing_stats['batch_processing_time']
        total_frames = self.processing_stats['total_frames_processed']
        
        performance_metrics = {
            'total_frames_processed': total_frames,
            'total_batches_submitted': self.processing_stats['total_batches_submitted'],
            'average_batch_size': self.processing_stats['average_batch_size'],
            'total_processing_time': total_time,
            'frames_per_second': total_frames / total_time if total_time > 0 else 0,
            'estimated_cost_savings': self._calculate_cost_savings(),
            'batching_efficiency': self._calculate_batching_efficiency()
        }
        
        logger.info(f"Batch processing finalized: {performance_metrics}")
        
        return {
            'performance_metrics': performance_metrics,
            'segment_results': dict(self.completed_analyses)
        }
    
    def _calculate_cost_savings(self) -> Dict[str, float]:
        """Calculate estimated cost savings from batch processing."""
        total_requests = self.processing_stats['total_frames_processed']
        batch_count = self.processing_stats['total_batches_submitted']
        
        # Gemini batch pricing is 50% of standard pricing
        individual_cost_multiplier = 1.0
        batch_cost_multiplier = 0.5
        
        individual_cost = total_requests * individual_cost_multiplier
        batch_cost = batch_count * batch_cost_multiplier  # Simplified calculation
        
        return {
            'estimated_individual_cost_units': individual_cost,
            'estimated_batch_cost_units': batch_cost,
            'estimated_savings_units': individual_cost - batch_cost,
            'estimated_savings_percentage': ((individual_cost - batch_cost) / individual_cost * 100) if individual_cost > 0 else 0
        }
    
    def _calculate_batching_efficiency(self) -> float:
        """
        Calculate batching efficiency based on how well requests were grouped.
        
        Returns:
            Efficiency score from 0.0 to 1.0
        """
        if self.processing_stats['total_batches_submitted'] == 0:
            return 0.0
        
        average_batch_size = self.processing_stats['average_batch_size']
        optimal_batch_size = self.batch_size
        
        # Efficiency is the ratio of actual vs optimal batch size
        efficiency = min(1.0, average_batch_size / optimal_batch_size)
        
        return efficiency


class CrossSegmentBatchCoordinator:
    """
    Coordinator for managing batch processing across multiple video segments.
    
    This class orchestrates the collection and batching of frame analysis requests
    from multiple segments, ensuring optimal utilization of the Gemini batch API.
    """
    
    def __init__(self, progress_tracker: ProgressTracker = None):
        """Initialize the batch coordinator."""
        self.progress_tracker = progress_tracker
        self.batch_processor = VisualBatchProcessor(progress_tracker)
        self.segment_frame_counts = {}
        self.processing_started = False
    
    def register_segment_frames(self, segment_id: str, frame_count: int) -> None:
        """
        Register the expected number of frames for a segment.
        
        Args:
            segment_id: Unique identifier for the segment
            frame_count: Number of frames expected for this segment
        """
        self.segment_frame_counts[segment_id] = frame_count
        logger.debug(f"Registered segment {segment_id} with {frame_count} frames")
    
    def submit_frame_for_analysis(
        self,
        segment_id: str,
        frame_path: str,
        frame_number: int,
        timestamp: float,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Submit a frame for batch analysis.
        
        Args:
            segment_id: Unique identifier for the segment
            frame_path: Path to the frame image file
            frame_number: Frame number within the segment
            timestamp: Timestamp of the frame in the video
            prompt: Analysis prompt for Gemini
            metadata: Additional metadata for the frame
        """
        self.batch_processor.add_frame_request(
            frame_path=frame_path,
            segment_id=segment_id,
            frame_number=frame_number,
            timestamp=timestamp,
            prompt=prompt,
            metadata=metadata
        )
        
        # Process batch if it's ready
        if self.batch_processor.should_flush_batch():
            batch_results = self.batch_processor.process_pending_batch()
            self.batch_processor.completed_analyses.update(batch_results)
    
    def get_segment_analyses(self, segment_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve completed analyses for a specific segment.
        
        Args:
            segment_id: The segment identifier
            
        Returns:
            List of frame analysis results for the segment
        """
        return self.batch_processor.get_segment_results(segment_id)
    
    def finalize_and_get_results(self) -> Dict[str, Any]:
        """
        Complete all pending processing and return final results.
        
        Returns:
            Dictionary containing all segment results and performance metrics
        """
        final_results = self.batch_processor.finalize_processing()
        
        # Log performance summary
        metrics = final_results['performance_metrics']
        logger.info(
            f"Cross-segment batch processing completed:\n"
            f"  - Total frames: {metrics['total_frames_processed']}\n"
            f"  - Total batches: {metrics['total_batches_submitted']}\n"
            f"  - Average batch size: {metrics['average_batch_size']:.1f}\n"
            f"  - Processing rate: {metrics['frames_per_second']:.2f} frames/second\n"
            f"  - Estimated cost savings: {metrics['estimated_cost_savings']['estimated_savings_percentage']:.1f}%\n"
            f"  - Batching efficiency: {metrics['batching_efficiency']*100:.1f}%"
        )
        
        return final_results