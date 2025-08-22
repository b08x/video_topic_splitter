#!/usr/bin/env python3
"""
Performance monitoring and metrics collection for batch processing optimizations.

This module provides real-time performance monitoring and metrics collection
to track the effectiveness of batch processing optimizations in production.
"""

import time
import json
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation."""
    timestamp: float
    operation_type: str  # 'individual', 'batch', 'hybrid'
    frames_processed: int
    api_calls_made: int
    execution_time: float
    error_count: int
    success_rate: float
    cost_efficiency_factor: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PerformanceMonitor:
    """
    Real-time performance monitoring for visual analysis operations.
    
    This class collects and analyzes performance metrics from batch processing
    operations, providing insights into optimization effectiveness and identifying
    potential bottlenecks or regressions.
    """
    
    def __init__(self, project_path: str = None, enable_detailed_logging: bool = False):
        """
        Initialize the performance monitor.
        
        Args:
            project_path: Project directory for saving metrics (optional)
            enable_detailed_logging: Enable detailed performance logging
        """
        self.project_path = project_path
        self.enable_detailed_logging = enable_detailed_logging
        self.metrics_buffer: List[ProcessingMetrics] = []
        self.session_start_time = time.time()
        self.cumulative_stats = defaultdict(list)
    
    def record_processing_operation(
        self,
        operation_type: str,
        frames_processed: int,
        api_calls_made: int,
        execution_time: float,
        error_count: int = 0,
        cost_efficiency_factor: float = 1.0
    ) -> None:
        """
        Record a processing operation for performance tracking.
        
        Args:
            operation_type: Type of processing ('individual', 'batch', 'hybrid')
            frames_processed: Number of frames processed
            api_calls_made: Number of API calls made
            execution_time: Total execution time in seconds
            error_count: Number of errors encountered
            cost_efficiency_factor: Cost efficiency factor (2.0 for 50% savings)
        """
        success_rate = (frames_processed - error_count) / frames_processed if frames_processed > 0 else 0
        
        metrics = ProcessingMetrics(
            timestamp=time.time(),
            operation_type=operation_type,
            frames_processed=frames_processed,
            api_calls_made=api_calls_made,
            execution_time=execution_time,
            error_count=error_count,
            success_rate=success_rate,
            cost_efficiency_factor=cost_efficiency_factor
        )
        
        self.metrics_buffer.append(metrics)
        self.cumulative_stats[operation_type].append(metrics)
        
        if self.enable_detailed_logging:
            logger.info(
                f"Performance recorded - {operation_type}: "
                f"{frames_processed} frames in {execution_time:.2f}s "
                f"({frames_processed/execution_time:.2f} fps), "
                f"{api_calls_made} API calls"
            )
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get current session performance metrics.
        
        Returns:
            Dictionary containing real-time performance data
        """
        if not self.metrics_buffer:
            return {"error": "No metrics collected yet"}
        
        session_duration = time.time() - self.session_start_time
        total_frames = sum(m.frames_processed for m in self.metrics_buffer)
        total_api_calls = sum(m.api_calls_made for m in self.metrics_buffer)
        total_errors = sum(m.error_count for m in self.metrics_buffer)
        
        # Calculate averages by operation type
        operation_stats = {}
        for op_type, metrics_list in self.cumulative_stats.items():
            if metrics_list:
                avg_execution_time = sum(m.execution_time for m in metrics_list) / len(metrics_list)
                avg_frames_per_op = sum(m.frames_processed for m in metrics_list) / len(metrics_list)
                avg_api_calls = sum(m.api_calls_made for m in metrics_list) / len(metrics_list)
                avg_success_rate = sum(m.success_rate for m in metrics_list) / len(metrics_list)
                
                operation_stats[op_type] = {
                    'operation_count': len(metrics_list),
                    'average_execution_time': avg_execution_time,
                    'average_frames_per_operation': avg_frames_per_op,
                    'average_api_calls': avg_api_calls,
                    'average_success_rate': avg_success_rate,
                    'total_frames': sum(m.frames_processed for m in metrics_list),
                    'throughput': avg_frames_per_op / avg_execution_time if avg_execution_time > 0 else 0
                }
        
        real_time_data = {
            'session_duration': session_duration,
            'total_operations': len(self.metrics_buffer),
            'total_frames_processed': total_frames,
            'total_api_calls': total_api_calls,
            'total_errors': total_errors,
            'overall_success_rate': (total_frames - total_errors) / total_frames if total_frames > 0 else 0,
            'overall_throughput': total_frames / session_duration if session_duration > 0 else 0,
            'api_efficiency': total_frames / total_api_calls if total_api_calls > 0 else 0,
            'operation_breakdown': operation_stats,
            'last_updated': time.time()
        }
        
        return real_time_data
    
    def calculate_optimization_impact(self) -> Dict[str, Any]:
        """
        Calculate the impact of batch processing optimizations.
        
        Returns:
            Dictionary containing optimization impact metrics
        """
        individual_metrics = self.cumulative_stats.get('individual', [])
        batch_metrics = self.cumulative_stats.get('batch', [])
        
        if not individual_metrics or not batch_metrics:
            return {
                "warning": "Insufficient data for optimization impact calculation",
                "individual_operations": len(individual_metrics),
                "batch_operations": len(batch_metrics)
            }
        
        # Calculate averages
        def calc_averages(metrics_list):
            if not metrics_list:
                return None
            return {
                'avg_execution_time': sum(m.execution_time for m in metrics_list) / len(metrics_list),
                'avg_frames_per_second': sum(m.frames_processed / m.execution_time for m in metrics_list if m.execution_time > 0) / len(metrics_list),
                'avg_api_efficiency': sum(m.frames_processed / m.api_calls_made for m in metrics_list if m.api_calls_made > 0) / len(metrics_list),
                'avg_success_rate': sum(m.success_rate for m in metrics_list) / len(metrics_list)
            }
        
        individual_avg = calc_averages(individual_metrics)
        batch_avg = calc_averages(batch_metrics)
        
        if not individual_avg or not batch_avg:
            return {"error": "Could not calculate averages"}
        
        # Calculate improvement ratios
        impact = {
            'performance_improvements': {
                'speed_improvement_factor': batch_avg['avg_frames_per_second'] / individual_avg['avg_frames_per_second'] if individual_avg['avg_frames_per_second'] > 0 else 0,
                'api_efficiency_improvement': batch_avg['avg_api_efficiency'] / individual_avg['avg_api_efficiency'] if individual_avg['avg_api_efficiency'] > 0 else 0,
                'cost_savings_percentage': 50.0,  # Gemini batch mode savings
                'latency_reduction_factor': individual_avg['avg_execution_time'] / batch_avg['avg_execution_time'] if batch_avg['avg_execution_time'] > 0 else 0
            },
            'reliability_metrics': {
                'individual_success_rate': individual_avg['avg_success_rate'],
                'batch_success_rate': batch_avg['avg_success_rate'],
                'reliability_change': batch_avg['avg_success_rate'] - individual_avg['avg_success_rate']
            },
            'sample_sizes': {
                'individual_operations': len(individual_metrics),
                'batch_operations': len(batch_metrics)
            }
        }
        
        return impact
    
    def save_metrics_report(self, filename: str = None) -> str:
        """
        Save collected metrics to a JSON report file.
        
        Args:
            filename: Optional custom filename for the report
            
        Returns:
            Path to the saved report file
        """
        if not self.project_path:
            logger.warning("No project path configured, metrics report not saved")
            return ""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        report_path = os.path.join(self.project_path, filename)
        
        report_data = {
            'collection_metadata': {
                'session_start_time': self.session_start_time,
                'report_generation_time': time.time(),
                'session_duration': time.time() - self.session_start_time,
                'total_operations': len(self.metrics_buffer)
            },
            'real_time_metrics': self.get_real_time_metrics(),
            'optimization_impact': self.calculate_optimization_impact(),
            'detailed_metrics': [m.to_dict() for m in self.metrics_buffer]
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Performance metrics report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to save metrics report: {e}")
            return ""
    
    def get_performance_recommendations(self) -> List[str]:
        """
        Generate performance optimization recommendations based on collected metrics.
        
        Returns:
            List of actionable performance recommendations
        """
        recommendations = []
        
        real_time_data = self.get_real_time_metrics()
        
        if 'operation_breakdown' not in real_time_data:
            return ["Insufficient data for recommendations"]
        
        operation_breakdown = real_time_data['operation_breakdown']
        
        # Analyze batch processing usage
        batch_stats = operation_breakdown.get('batch', {})
        individual_stats = operation_breakdown.get('individual', {})
        
        if batch_stats and individual_stats:
            batch_throughput = batch_stats.get('throughput', 0)
            individual_throughput = individual_stats.get('throughput', 0)
            
            if batch_throughput > individual_throughput * 1.5:
                recommendations.append(
                    f"Batch processing is highly effective ({batch_throughput/individual_throughput:.1f}x improvement). "
                    "Consider increasing batch size for even better performance."
                )
            elif batch_throughput > individual_throughput:
                recommendations.append(
                    "Batch processing provides moderate improvements. "
                    "Monitor for consistent gains and consider tuning batch size."
                )
            else:
                recommendations.append(
                    "Batch processing may not be optimal for current workload. "
                    "Consider reviewing batch size configuration or network conditions."
                )
        
        # Analyze API efficiency
        overall_api_efficiency = real_time_data.get('api_efficiency', 0)
        if overall_api_efficiency < 5:
            recommendations.append(
                f"Low API efficiency detected ({overall_api_efficiency:.1f} frames/call). "
                "Consider increasing batch sizes or reducing frame extraction frequency."
            )
        elif overall_api_efficiency > 20:
            recommendations.append(
                f"Excellent API efficiency ({overall_api_efficiency:.1f} frames/call). "
                "Current optimization strategy is highly effective."
            )
        
        # Analyze error rates
        overall_success_rate = real_time_data.get('overall_success_rate', 0)
        if overall_success_rate < 0.9:
            recommendations.append(
                f"Success rate is below optimal ({overall_success_rate*100:.1f}%). "
                "Consider implementing additional error handling or retry logic."
            )
        
        # Analyze session throughput
        overall_throughput = real_time_data.get('overall_throughput', 0)
        if overall_throughput < 1.0:
            recommendations.append(
                f"Low overall throughput ({overall_throughput:.2f} frames/second). "
                "Consider optimizing batch sizes, network configuration, or parallel processing."
            )
        
        return recommendations


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(project_path: str = None) -> PerformanceMonitor:
    """
    Get or create the global performance monitor instance.
    
    Args:
        project_path: Project directory for saving metrics
        
    Returns:
        Global PerformanceMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        from .batch_config import get_batch_config
        config = get_batch_config()
        
        _global_monitor = PerformanceMonitor(
            project_path=project_path,
            enable_detailed_logging=config.log_detailed_metrics
        )
    
    return _global_monitor


def record_visual_analysis_metrics(
    operation_type: str,
    frames_processed: int,
    execution_time: float,
    api_calls_made: int = None,
    error_count: int = 0
) -> None:
    """
    Convenience function to record visual analysis performance metrics.
    
    Args:
        operation_type: Type of processing operation
        frames_processed: Number of frames processed
        execution_time: Total execution time
        api_calls_made: Number of API calls (estimated if None)
        error_count: Number of errors encountered
    """
    monitor = get_performance_monitor()
    
    # Estimate API calls if not provided
    if api_calls_made is None:
        if operation_type == 'batch':
            # Estimate based on typical batch size
            from .batch_config import get_batch_config
            config = get_batch_config()
            api_calls_made = max(1, (frames_processed + config.optimal_batch_size - 1) // config.optimal_batch_size)
        else:
            api_calls_made = frames_processed  # Individual processing
    
    # Calculate cost efficiency factor
    cost_efficiency_factor = 2.0 if operation_type == 'batch' else 1.0
    
    monitor.record_processing_operation(
        operation_type=operation_type,
        frames_processed=frames_processed,
        api_calls_made=api_calls_made,
        execution_time=execution_time,
        error_count=error_count,
        cost_efficiency_factor=cost_efficiency_factor
    )


def generate_session_performance_report(project_path: str = None) -> Dict[str, Any]:
    """
    Generate a comprehensive performance report for the current session.
    
    Args:
        project_path: Project directory for saving the report
        
    Returns:
        Dictionary containing the performance report
    """
    monitor = get_performance_monitor(project_path)
    
    # Generate and save the report
    if project_path:
        report_path = monitor.save_metrics_report()
        
        # Also generate recommendations
        recommendations = monitor.get_performance_recommendations()
        
        # Save recommendations separately
        if recommendations:
            rec_path = os.path.join(project_path, "performance_recommendations.json")
            try:
                with open(rec_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_time': time.time(),
                        'recommendations': recommendations,
                        'metrics_source': report_path
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Performance recommendations saved to: {rec_path}")
            except Exception as e:
                logger.warning(f"Could not save recommendations: {e}")
    
    # Return current metrics
    return {
        'real_time_metrics': monitor.get_real_time_metrics(),
        'optimization_impact': monitor.calculate_optimization_impact(),
        'recommendations': monitor.get_performance_recommendations()
    }


def enable_performance_monitoring(project_path: str = None) -> None:
    """
    Enable global performance monitoring for the current session.
    
    Args:
        project_path: Project directory for saving metrics and reports
    """
    global _global_monitor
    
    from .batch_config import get_batch_config
    config = get_batch_config()
    
    if config.enable_performance_logging:
        _global_monitor = PerformanceMonitor(
            project_path=project_path,
            enable_detailed_logging=config.log_detailed_metrics
        )
        
        logger.info("Performance monitoring enabled for visual analysis optimization")
    else:
        logger.debug("Performance monitoring disabled by configuration")


def disable_performance_monitoring() -> Optional[Dict[str, Any]]:
    """
    Disable performance monitoring and return final metrics.
    
    Returns:
        Final performance metrics if monitoring was active
    """
    global _global_monitor
    
    if _global_monitor:
        final_metrics = _global_monitor.get_real_time_metrics()
        _global_monitor = None
        logger.info("Performance monitoring disabled")
        return final_metrics
    
    return None