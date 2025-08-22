#!/usr/bin/env python3
"""
Configuration management for batch processing optimizations.

This module provides configuration parameters and utilities for tuning
the performance of batch processing operations in the video topic splitter.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class BatchProcessingConfig:
    """Configuration parameters for batch processing optimization."""
    
    # Batch size configuration
    max_batch_size: int = 50
    min_batch_size: int = 5
    optimal_batch_size: int = 25
    
    # Timing configuration
    max_batch_wait_time: int = 300  # 5 minutes
    poll_interval: int = 5          # 5 seconds
    
    # Quality configuration
    frame_extraction_quality: int = 90
    frames_per_segment: int = 3
    
    # Fallback configuration
    enable_fallback_to_individual: bool = True
    max_retry_attempts: int = 2
    
    # Performance monitoring
    enable_performance_logging: bool = True
    log_detailed_metrics: bool = False
    
    @classmethod
    def from_environment(cls) -> 'BatchProcessingConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            BatchProcessingConfig instance with values from environment
        """
        return cls(
            max_batch_size=int(os.getenv('GEMINI_MAX_BATCH_SIZE', 50)),
            min_batch_size=int(os.getenv('GEMINI_MIN_BATCH_SIZE', 5)),
            optimal_batch_size=int(os.getenv('GEMINI_OPTIMAL_BATCH_SIZE', 25)),
            max_batch_wait_time=int(os.getenv('GEMINI_MAX_WAIT_TIME', 300)),
            poll_interval=int(os.getenv('GEMINI_POLL_INTERVAL', 5)),
            frame_extraction_quality=int(os.getenv('FRAME_QUALITY', 90)),
            frames_per_segment=int(os.getenv('FRAMES_PER_SEGMENT', 3)),
            enable_fallback_to_individual=os.getenv('ENABLE_BATCH_FALLBACK', 'true').lower() == 'true',
            max_retry_attempts=int(os.getenv('MAX_RETRY_ATTEMPTS', 2)),
            enable_performance_logging=os.getenv('ENABLE_PERF_LOGGING', 'true').lower() == 'true',
            log_detailed_metrics=os.getenv('LOG_DETAILED_METRICS', 'false').lower() == 'true'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size,
            'optimal_batch_size': self.optimal_batch_size,
            'max_batch_wait_time': self.max_batch_wait_time,
            'poll_interval': self.poll_interval,
            'frame_extraction_quality': self.frame_extraction_quality,
            'frames_per_segment': self.frames_per_segment,
            'enable_fallback_to_individual': self.enable_fallback_to_individual,
            'max_retry_attempts': self.max_retry_attempts,
            'enable_performance_logging': self.enable_performance_logging,
            'log_detailed_metrics': self.log_detailed_metrics
        }
    
    def validate(self) -> Dict[str, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}
        
        if self.max_batch_size < self.min_batch_size:
            errors['batch_size'] = 'max_batch_size must be >= min_batch_size'
        
        if self.optimal_batch_size < self.min_batch_size or self.optimal_batch_size > self.max_batch_size:
            errors['optimal_batch_size'] = 'optimal_batch_size must be between min and max batch sizes'
        
        if self.max_batch_wait_time <= 0:
            errors['max_batch_wait_time'] = 'max_batch_wait_time must be positive'
        
        if self.poll_interval <= 0:
            errors['poll_interval'] = 'poll_interval must be positive'
        
        if self.frame_extraction_quality < 10 or self.frame_extraction_quality > 100:
            errors['frame_extraction_quality'] = 'frame_extraction_quality must be between 10 and 100'
        
        if self.frames_per_segment < 1:
            errors['frames_per_segment'] = 'frames_per_segment must be at least 1'
        
        return errors


# Global configuration instance
default_config = BatchProcessingConfig.from_environment()


def get_batch_config() -> BatchProcessingConfig:
    """Get the current batch processing configuration."""
    return default_config


def update_batch_config(**kwargs) -> BatchProcessingConfig:
    """
    Update batch processing configuration with new values.
    
    Args:
        **kwargs: Configuration parameters to update
        
    Returns:
        Updated configuration instance
    """
    global default_config
    
    config_dict = default_config.to_dict()
    config_dict.update(kwargs)
    
    default_config = BatchProcessingConfig(**config_dict)
    
    # Validate updated configuration
    errors = default_config.validate()
    if errors:
        raise ValueError(f"Invalid configuration parameters: {errors}")
    
    return default_config


def optimize_batch_size_for_workload(
    total_frames: int, 
    available_memory_mb: int = 1024
) -> int:
    """
    Calculate optimal batch size based on workload characteristics.
    
    Args:
        total_frames: Total number of frames to process
        available_memory_mb: Available memory in MB for batch processing
        
    Returns:
        Recommended batch size
    """
    config = get_batch_config()
    
    # Memory-based calculation (rough estimate: ~2MB per high-quality frame)
    memory_based_limit = available_memory_mb // 2
    
    # Workload-based calculation
    if total_frames <= config.min_batch_size:
        workload_optimal = total_frames
    elif total_frames <= config.optimal_batch_size * 2:
        workload_optimal = config.optimal_batch_size
    else:
        # For large workloads, aim for multiple reasonably-sized batches
        workload_optimal = min(config.max_batch_size, total_frames // 3)
    
    # Take the minimum of memory and workload constraints
    recommended_size = min(
        config.max_batch_size,
        max(config.min_batch_size, min(memory_based_limit, workload_optimal))
    )
    
    return recommended_size