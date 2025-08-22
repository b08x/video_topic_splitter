#!/usr/bin/env python3
"""
Performance testing utilities for visual analysis optimizations.

This module provides tools to measure and validate the performance improvements
from batch processing optimizations in the video topic splitter.
"""

import time
import logging
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    operation_name: str
    execution_time: float
    api_calls_count: int
    frames_processed: int
    cost_efficiency: float
    error_count: int
    
    @property
    def frames_per_second(self) -> float:
        """Calculate processing rate in frames per second."""
        return self.frames_processed / self.execution_time if self.execution_time > 0 else 0
    
    @property
    def api_efficiency(self) -> float:
        """Calculate API efficiency (frames per API call)."""
        return self.frames_processed / self.api_calls_count if self.api_calls_count > 0 else 0


class PerformanceBenchmark:
    """
    Performance benchmarking utility for visual analysis optimization validation.
    
    This class provides tools to measure and compare the performance of
    individual vs batch processing approaches for Gemini API requests.
    """
    
    def __init__(self):
        """Initialize the performance benchmark."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """
        Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            
        Yields:
            Dictionary to collect measurement data during operation
        """
        measurement_data = {
            'api_calls_count': 0,
            'frames_processed': 0,
            'error_count': 0,
            'cost_efficiency': 1.0
        }
        
        start_time = time.time()
        
        try:
            yield measurement_data
        finally:
            execution_time = time.time() - start_time
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                api_calls_count=measurement_data['api_calls_count'],
                frames_processed=measurement_data['frames_processed'],
                cost_efficiency=measurement_data['cost_efficiency'],
                error_count=measurement_data['error_count']
            )
            
            self.metrics_history.append(metrics)
            
            if operation_name == 'baseline':
                self.baseline_metrics = metrics
            
            logger.info(
                f"Performance measurement '{operation_name}' completed:\n"
                f"  - Execution time: {execution_time:.2f}s\n"
                f"  - Frames processed: {measurement_data['frames_processed']}\n"
                f"  - API calls: {measurement_data['api_calls_count']}\n"
                f"  - Frames/second: {metrics.frames_per_second:.2f}\n"
                f"  - API efficiency: {metrics.api_efficiency:.2f} frames/call"
            )
    
    def compare_with_baseline(self, test_metrics: PerformanceMetrics) -> Dict[str, float]:
        """
        Compare test metrics with baseline performance.
        
        Args:
            test_metrics: Metrics from the optimized test run
            
        Returns:
            Dictionary containing comparison ratios and improvements
        """
        if not self.baseline_metrics:
            logger.warning("No baseline metrics available for comparison")
            return {}
        
        baseline = self.baseline_metrics
        
        comparison = {
            'speed_improvement_ratio': baseline.execution_time / test_metrics.execution_time if test_metrics.execution_time > 0 else 0,
            'api_efficiency_improvement': test_metrics.api_efficiency / baseline.api_efficiency if baseline.api_efficiency > 0 else 0,
            'cost_efficiency_improvement': test_metrics.cost_efficiency / baseline.cost_efficiency if baseline.cost_efficiency > 0 else 0,
            'error_rate_change': test_metrics.error_count - baseline.error_count,
            'throughput_improvement': test_metrics.frames_per_second / baseline.frames_per_second if baseline.frames_per_second > 0 else 0
        }
        
        logger.info(
            f"Performance comparison vs baseline:\n"
            f"  - Speed improvement: {comparison['speed_improvement_ratio']:.2f}x\n"
            f"  - API efficiency: {comparison['api_efficiency_improvement']:.2f}x\n"
            f"  - Cost efficiency: {comparison['cost_efficiency_improvement']:.2f}x\n"
            f"  - Throughput improvement: {comparison['throughput_improvement']:.2f}x\n"
            f"  - Error rate change: {comparison['error_rate_change']:+d}"
        )
        
        return comparison
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing detailed performance analysis
        """
        if not self.metrics_history:
            return {"error": "No performance metrics collected"}
        
        # Calculate summary statistics
        execution_times = [m.execution_time for m in self.metrics_history]
        api_calls = [m.api_calls_count for m in self.metrics_history]
        frames_processed = [m.frames_processed for m in self.metrics_history]
        
        report = {
            'summary': {
                'total_operations': len(self.metrics_history),
                'total_execution_time': sum(execution_times),
                'total_frames_processed': sum(frames_processed),
                'total_api_calls': sum(api_calls),
                'average_execution_time': statistics.mean(execution_times),
                'average_frames_per_operation': statistics.mean(frames_processed) if frames_processed else 0,
                'overall_throughput': sum(frames_processed) / sum(execution_times) if sum(execution_times) > 0 else 0
            },
            'operations': [
                {
                    'name': m.operation_name,
                    'execution_time': m.execution_time,
                    'frames_processed': m.frames_processed,
                    'api_calls': m.api_calls_count,
                    'frames_per_second': m.frames_per_second,
                    'api_efficiency': m.api_efficiency,
                    'error_count': m.error_count
                }
                for m in self.metrics_history
            ]
        }
        
        # Add comparison if baseline exists
        if self.baseline_metrics:
            optimized_metrics = [m for m in self.metrics_history if m.operation_name != 'baseline']
            if optimized_metrics:
                best_optimized = max(optimized_metrics, key=lambda x: x.frames_per_second)
                report['optimization_analysis'] = self.compare_with_baseline(best_optimized)
        
        return report


class APICallCounter:
    """
    Utility class to track API call patterns and performance.
    
    This can be used to instrument existing code to measure the impact
    of batch processing optimizations.
    """
    
    def __init__(self):
        """Initialize the API call counter."""
        self.call_history: List[Dict[str, Any]] = []
        self.active_batch: Optional[str] = None
    
    def record_api_call(
        self, 
        call_type: str, 
        request_count: int = 1, 
        response_time: float = 0, 
        success: bool = True
    ) -> None:
        """
        Record an API call for performance tracking.
        
        Args:
            call_type: Type of API call ('individual' or 'batch')
            request_count: Number of requests in this call
            response_time: Time taken for the API call
            success: Whether the call was successful
        """
        call_record = {
            'timestamp': time.time(),
            'call_type': call_type,
            'request_count': request_count,
            'response_time': response_time,
            'success': success,
            'batch_id': self.active_batch
        }
        
        self.call_history.append(call_record)
        
        logger.debug(f"Recorded {call_type} API call: {request_count} requests in {response_time:.3f}s")
    
    def start_batch_tracking(self, batch_id: str) -> None:
        """Start tracking calls for a specific batch."""
        self.active_batch = batch_id
        logger.debug(f"Started tracking batch: {batch_id}")
    
    def end_batch_tracking(self) -> None:
        """End current batch tracking."""
        if self.active_batch:
            logger.debug(f"Ended tracking batch: {self.active_batch}")
        self.active_batch = None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate performance summary from recorded API calls.
        
        Returns:
            Dictionary containing API call performance statistics
        """
        if not self.call_history:
            return {"error": "No API calls recorded"}
        
        individual_calls = [call for call in self.call_history if call['call_type'] == 'individual']
        batch_calls = [call for call in self.call_history if call['call_type'] == 'batch']
        
        summary = {
            'total_calls': len(self.call_history),
            'individual_calls': {
                'count': len(individual_calls),
                'total_requests': sum(call['request_count'] for call in individual_calls),
                'total_time': sum(call['response_time'] for call in individual_calls),
                'average_response_time': statistics.mean([call['response_time'] for call in individual_calls]) if individual_calls else 0,
                'success_rate': sum(1 for call in individual_calls if call['success']) / len(individual_calls) if individual_calls else 0
            },
            'batch_calls': {
                'count': len(batch_calls),
                'total_requests': sum(call['request_count'] for call in batch_calls),
                'total_time': sum(call['response_time'] for call in batch_calls),
                'average_response_time': statistics.mean([call['response_time'] for call in batch_calls]) if batch_calls else 0,
                'average_batch_size': statistics.mean([call['request_count'] for call in batch_calls]) if batch_calls else 0,
                'success_rate': sum(1 for call in batch_calls if call['success']) / len(batch_calls) if batch_calls else 0
            }
        }
        
        # Calculate efficiency metrics
        if individual_calls and batch_calls:
            individual_throughput = summary['individual_calls']['total_requests'] / summary['individual_calls']['total_time'] if summary['individual_calls']['total_time'] > 0 else 0
            batch_throughput = summary['batch_calls']['total_requests'] / summary['batch_calls']['total_time'] if summary['batch_calls']['total_time'] > 0 else 0
            
            summary['efficiency_comparison'] = {
                'individual_throughput': individual_throughput,
                'batch_throughput': batch_throughput,
                'throughput_improvement': batch_throughput / individual_throughput if individual_throughput > 0 else 0,
                'estimated_cost_savings': 50.0 if batch_calls else 0.0  # Gemini batch discount
            }
        
        return summary


# Global API call counter for instrumentation
api_call_counter = APICallCounter()


def instrument_gemini_calls():
    """
    Instrument Gemini API calls for performance measurement.
    
    This function can be called to add performance tracking to existing
    Gemini API calls without modifying the core logic.
    """
    from ..api.gemini import analyze_with_gemini, batch_analyze_images_with_gemini
    
    # Monkey patch individual calls
    original_analyze = analyze_with_gemini
    
    def instrumented_analyze(prompt, image=None):
        start_time = time.time()
        try:
            result = original_analyze(prompt, image)
            response_time = time.time() - start_time
            api_call_counter.record_api_call('individual', 1, response_time, True)
            return result
        except Exception as e:
            response_time = time.time() - start_time
            api_call_counter.record_api_call('individual', 1, response_time, False)
            raise
    
    # Monkey patch batch calls
    original_batch_analyze = batch_analyze_images_with_gemini
    
    def instrumented_batch_analyze(image_analysis_requests, progress_callback=None):
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}"
        api_call_counter.start_batch_tracking(batch_id)
        
        try:
            result = original_batch_analyze(image_analysis_requests, progress_callback)
            response_time = time.time() - start_time
            api_call_counter.record_api_call('batch', len(image_analysis_requests), response_time, True)
            return result
        except Exception as e:
            response_time = time.time() - start_time
            api_call_counter.record_api_call('batch', len(image_analysis_requests), response_time, False)
            raise
        finally:
            api_call_counter.end_batch_tracking()
    
    # Replace functions (for testing purposes)
    return instrumented_analyze, instrumented_batch_analyze


def run_performance_comparison_test(
    video_path: str,
    segment_list: List[Dict[str, Any]],
    transcript_data: List[Dict[str, Any]],
    progress_tracker=None
) -> Dict[str, Any]:
    """
    Run a comprehensive performance comparison between individual and batch processing.
    
    Args:
        video_path: Path to the video file
        segment_list: List of video segments to process
        transcript_data: Transcript data for the video
        progress_tracker: Optional progress tracker
        
    Returns:
        Dictionary containing detailed performance comparison results
    """
    from .multimodal_analysis import MultimodalAnalyzer
    from .segment_analysis import SegmentProcessor
    
    logger.info("Starting performance comparison test")
    
    benchmark = PerformanceBenchmark()
    
    # Test 1: Individual processing (baseline)
    logger.info("Testing individual processing (baseline)")
    
    with benchmark.measure_operation('individual_processing') as baseline_measurement:
        processor_individual = SegmentProcessor(
            progress_tracker=progress_tracker, 
            enable_batch_optimization=False
        )
        
        start_time = time.time()
        individual_results = processor_individual.process_segments(
            video_path, segment_list[:3], transcript_data  # Test with first 3 segments
        )
        
        baseline_measurement['frames_processed'] = sum(
            result.get('visual_analysis', {}).get('frames_extracted', 0) 
            for result in individual_results
        )
        baseline_measurement['api_calls_count'] = baseline_measurement['frames_processed']  # 1 call per frame
        baseline_measurement['cost_efficiency'] = 1.0  # Standard pricing
    
    # Test 2: Batch processing (optimized)
    logger.info("Testing batch processing (optimized)")
    
    with benchmark.measure_operation('batch_processing') as batch_measurement:
        start_time = time.time()
        batch_results = MultimodalAnalyzer.analyze_multiple_segments_optimized(
            video_path, segment_list[:3], transcript_data, progress_tracker
        )
        
        batch_measurement['frames_processed'] = sum(
            result.get('visual_analysis', {}).get('frames_extracted', 0) 
            for result in batch_results
        )
        
        # Estimate batch call count (assuming optimal batching)
        total_frames = batch_measurement['frames_processed']
        batch_size = 50  # Default batch size
        batch_measurement['api_calls_count'] = max(1, (total_frames + batch_size - 1) // batch_size)
        batch_measurement['cost_efficiency'] = 2.0  # 50% cost savings = 2x efficiency
    
    # Generate comparison report
    comparison_results = benchmark.generate_performance_report()
    
    # Add specific optimization metrics
    if baseline_measurement['frames_processed'] > 0 and batch_measurement['frames_processed'] > 0:
        optimization_impact = {
            'latency_reduction': {
                'baseline_time': benchmark.baseline_metrics.execution_time,
                'optimized_time': benchmark.metrics_history[-1].execution_time,
                'improvement_factor': benchmark.baseline_metrics.execution_time / benchmark.metrics_history[-1].execution_time
            },
            'api_call_reduction': {
                'baseline_calls': baseline_measurement['api_calls_count'],
                'optimized_calls': batch_measurement['api_calls_count'],
                'reduction_percentage': (1 - batch_measurement['api_calls_count'] / baseline_measurement['api_calls_count']) * 100
            },
            'cost_optimization': {
                'estimated_cost_reduction_percentage': 50.0,  # Gemini batch pricing
                'processing_efficiency_gain': batch_measurement['cost_efficiency'] / baseline_measurement['cost_efficiency']
            }
        }
        
        comparison_results['optimization_impact'] = optimization_impact
    
    logger.info("Performance comparison test completed")
    return comparison_results


def validate_batch_processing_accuracy(
    individual_results: List[Dict[str, Any]],
    batch_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate that batch processing produces equivalent results to individual processing.
    
    Args:
        individual_results: Results from individual frame processing
        batch_results: Results from batch processing
        
    Returns:
        Dictionary containing accuracy validation metrics
    """
    logger.info("Validating batch processing accuracy")
    
    validation = {
        'segments_compared': 0,
        'frames_compared': 0,
        'content_similarity_scores': [],
        'processing_differences': [],
        'accuracy_summary': {}
    }
    
    try:
        # Compare segment by segment
        for i, (individual_seg, batch_seg) in enumerate(zip(individual_results, batch_results)):
            validation['segments_compared'] += 1
            
            # Compare visual analysis results
            individual_visual = individual_seg.get('visual_analysis', {})
            batch_visual = batch_seg.get('visual_analysis', {})
            
            individual_frames = individual_visual.get('frame_analyses', [])
            batch_frames = batch_visual.get('frame_analyses', [])
            
            # Compare frame analyses
            for j, (ind_frame, batch_frame) in enumerate(zip(individual_frames, batch_frames)):
                validation['frames_compared'] += 1
                
                ind_analysis = ind_frame.get('analysis', '')
                batch_analysis = batch_frame.get('analysis', '')
                
                # Simple similarity check (could be enhanced with more sophisticated NLP)
                similarity = _calculate_content_similarity(ind_analysis, batch_analysis)
                validation['content_similarity_scores'].append(similarity)
                
                if similarity < 0.8:  # Flag significant differences
                    validation['processing_differences'].append({
                        'segment': i + 1,
                        'frame': j + 1,
                        'similarity_score': similarity,
                        'individual_length': len(ind_analysis),
                        'batch_length': len(batch_analysis)
                    })
        
        # Calculate accuracy metrics
        if validation['content_similarity_scores']:
            avg_similarity = statistics.mean(validation['content_similarity_scores'])
            min_similarity = min(validation['content_similarity_scores'])
            
            validation['accuracy_summary'] = {
                'average_content_similarity': avg_similarity,
                'minimum_content_similarity': min_similarity,
                'high_accuracy_percentage': sum(1 for score in validation['content_similarity_scores'] if score >= 0.9) / len(validation['content_similarity_scores']) * 100,
                'acceptable_accuracy_percentage': sum(1 for score in validation['content_similarity_scores'] if score >= 0.8) / len(validation['content_similarity_scores']) * 100,
                'significant_differences_count': len(validation['processing_differences'])
            }
        
        logger.info(
            f"Accuracy validation completed:\n"
            f"  - Segments compared: {validation['segments_compared']}\n"
            f"  - Frames compared: {validation['frames_compared']}\n"
            f"  - Average similarity: {validation['accuracy_summary'].get('average_content_similarity', 0):.3f}\n"
            f"  - High accuracy rate: {validation['accuracy_summary'].get('high_accuracy_percentage', 0):.1f}%"
        )
        
    except Exception as e:
        logger.error(f"Error in accuracy validation: {e}")
        validation['error'] = str(e)
    
    return validation


def _calculate_content_similarity(text1: str, text2: str) -> float:
    """
    Calculate basic content similarity between two text strings.
    
    This uses a simple approach based on common words and length similarity.
    For production use, consider using more sophisticated NLP similarity measures.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0 if text1 != text2 else 1.0
    
    # Normalize texts
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    jaccard_similarity = intersection / union if union > 0 else 0
    
    # Factor in length similarity
    length_similarity = 1 - abs(len(text1) - len(text2)) / max(len(text1), len(text2))
    
    # Weighted combination
    combined_similarity = (jaccard_similarity * 0.7) + (length_similarity * 0.3)
    
    return min(1.0, max(0.0, combined_similarity))