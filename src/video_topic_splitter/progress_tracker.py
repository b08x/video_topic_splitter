#!/usr/bin/env python3
"""Centralized progress tracking system for video processing pipeline."""

import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from threading import Lock

try:
    import progressbar
    PROGRESSBAR_AVAILABLE = True
except ImportError:
    PROGRESSBAR_AVAILABLE = False

from .constants import CHECKPOINTS

logger = logging.getLogger(__name__)


@dataclass
class ProgressPhase:
    """Represents a phase in the processing pipeline."""
    name: str
    weight: float = 1.0
    description: str = ""
    current_progress: float = 0.0
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    sub_phases: List['ProgressPhase'] = None
    
    def __post_init__(self):
        if self.sub_phases is None:
            self.sub_phases = []


class ProgressTracker:
    """Centralized progress tracking for the video processing pipeline."""
    
    def __init__(self, project_path: str = None, json_output: bool = False):
        self.project_path = project_path
        self.json_output = json_output
        self.phases: List[ProgressPhase] = []
        self.current_phase_index = 0
        self.callbacks: List[Callable] = []
        self.lock = Lock()
        self.start_time = time.time()
        
        # Initialize default phases based on CHECKPOINTS
        self._initialize_default_phases()
    
    def _initialize_default_phases(self):
        """Initialize default phases based on the checkpoint system."""
        phase_configs = [
            ("Project Setup", 0.05, "Setting up project structure"),
            ("YouTube Download", 0.10, "Downloading video from YouTube"),
            ("Audio Processing", 0.15, "Processing and normalizing audio"),
            ("Transcription", 0.20, "Converting speech to text"),
            ("Topic Modeling", 0.25, "Analyzing topics and creating segments"),
            ("Scene Detection", 0.10, "Detecting visual scene changes"),
            ("Visual Analysis", 0.15, "Analyzing visual content and extracting frames"),
            ("Process Complete", 0.0, "Finalizing results")
        ]
        
        self.phases = [
            ProgressPhase(name=name, weight=weight, description=desc)
            for name, weight, desc in phase_configs
        ]
    
    def add_callback(self, callback: Callable):
        """Add a progress callback function."""
        self.callbacks.append(callback)
    
    def get_current_phase(self) -> Optional[ProgressPhase]:
        """Get the currently active phase."""
        if 0 <= self.current_phase_index < len(self.phases):
            return self.phases[self.current_phase_index]
        return None
    
    def get_overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        total_weight = sum(phase.weight for phase in self.phases)
        if total_weight == 0:
            return 0.0
        
        completed_weight = 0.0
        for i, phase in enumerate(self.phases):
            if phase.status == "completed":
                completed_weight += phase.weight
            elif phase.status == "in_progress" and i == self.current_phase_index:
                completed_weight += phase.weight * (phase.current_progress / 100.0)
        
        return min(100.0, (completed_weight / total_weight) * 100.0)
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """Estimate time remaining based on current progress."""
        overall_progress = self.get_overall_progress()
        if overall_progress <= 0:
            return None
        
        elapsed_time = time.time() - self.start_time
        total_estimated_time = elapsed_time * (100.0 / overall_progress)
        return max(0, total_estimated_time - elapsed_time)
    
    def start_phase(self, phase_name: str, checkpoint_stage: int = None):
        """Start a new phase."""
        with self.lock:
            # Find the phase by name
            for i, phase in enumerate(self.phases):
                if phase.name == phase_name:
                    self.current_phase_index = i
                    phase.status = "in_progress"
                    phase.start_time = time.time()
                    phase.current_progress = 0.0
                    break
            else:
                # Phase not found, create a new one
                new_phase = ProgressPhase(
                    name=phase_name,
                    weight=1.0,
                    description=f"Processing {phase_name}",
                    status="in_progress",
                    start_time=time.time()
                )
                self.phases.append(new_phase)
                self.current_phase_index = len(self.phases) - 1
            
            self._notify_callbacks()
    
    def update_phase_progress(self, progress: float, message: str = None):
        """Update progress for the current phase."""
        with self.lock:
            current_phase = self.get_current_phase()
            if current_phase:
                current_phase.current_progress = min(100.0, max(0.0, progress))
                if message:
                    current_phase.description = message
                self._notify_callbacks()
    
    def complete_phase(self, phase_name: str = None):
        """Mark a phase as completed."""
        with self.lock:
            if phase_name:
                # Complete specific phase
                for phase in self.phases:
                    if phase.name == phase_name:
                        phase.status = "completed"
                        phase.current_progress = 100.0
                        phase.end_time = time.time()
                        break
            else:
                # Complete current phase
                current_phase = self.get_current_phase()
                if current_phase:
                    current_phase.status = "completed"
                    current_phase.current_progress = 100.0
                    current_phase.end_time = time.time()
            
            self._notify_callbacks()
            
            # Print newline if this is the final phase
            if phase_name == "Process Complete" and not self.json_output:
                print()  # Clean line break after completion
    
    def fail_phase(self, error_message: str, phase_name: str = None):
        """Mark a phase as failed."""
        with self.lock:
            if phase_name:
                # Fail specific phase
                for phase in self.phases:
                    if phase.name == phase_name:
                        phase.status = "failed"
                        phase.description = f"Failed: {error_message}"
                        phase.end_time = time.time()
                        break
            else:
                # Fail current phase
                current_phase = self.get_current_phase()
                if current_phase:
                    current_phase.status = "failed"
                    current_phase.description = f"Failed: {error_message}"
                    current_phase.end_time = time.time()
            
            self._notify_callbacks()
    
    def add_sub_phase(self, parent_phase_name: str, sub_phase_name: str, weight: float = 1.0):
        """Add a sub-phase to a parent phase."""
        with self.lock:
            for phase in self.phases:
                if phase.name == parent_phase_name:
                    sub_phase = ProgressPhase(
                        name=sub_phase_name,
                        weight=weight,
                        description=f"Processing {sub_phase_name}"
                    )
                    phase.sub_phases.append(sub_phase)
                    break
    
    def _notify_callbacks(self):
        """Notify all registered callbacks of progress updates."""
        progress_data = {
            "overall_progress": self.get_overall_progress(),
            "current_phase": self.get_current_phase(),
            "estimated_time_remaining": self.get_estimated_time_remaining(),
            "phases": self.phases
        }
        
        for callback in self.callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
        
        # JSON output if requested
        if self.json_output:
            self._output_json_progress(progress_data)
    
    def _output_json_progress(self, progress_data: Dict):
        """Output progress as JSON for programmatic consumption."""
        # Convert dataclasses to dict for JSON serialization
        serializable_data = {
            "overall_progress": progress_data["overall_progress"],
            "estimated_time_remaining": progress_data["estimated_time_remaining"],
            "current_phase": asdict(progress_data["current_phase"]) if progress_data["current_phase"] else None,
            "phases": [asdict(phase) for phase in progress_data["phases"]]
        }
        
        print(f"PROGRESS_JSON: {json.dumps(serializable_data)}")
    
    @contextmanager
    def phase_context(self, phase_name: str, checkpoint_stage: int = None):
        """Context manager for phase execution."""
        self.start_phase(phase_name, checkpoint_stage)
        try:
            yield self
        except Exception as e:
            self.fail_phase(str(e))
            raise
        else:
            self.complete_phase()
    
    def create_sub_progress_tracker(self, parent_phase_name: str, sub_items: List[str]) -> 'SubProgressTracker':
        """Create a sub-progress tracker for detailed progress within a phase."""
        return SubProgressTracker(self, parent_phase_name, sub_items)


class SubProgressTracker:
    """Tracks progress for sub-items within a phase."""
    
    def __init__(self, parent_tracker: ProgressTracker, parent_phase_name: str, sub_items: List[str]):
        self.parent_tracker = parent_tracker
        self.parent_phase_name = parent_phase_name
        self.sub_items = sub_items
        self.current_item_index = 0
        self.current_item_progress = 0.0
    
    def update_item_progress(self, progress: float, message: str = None):
        """Update progress for the current sub-item."""
        self.current_item_progress = min(100.0, max(0.0, progress))
        
        # Calculate overall progress within the parent phase
        if len(self.sub_items) > 0:
            completed_items = self.current_item_index
            current_item_contribution = self.current_item_progress / 100.0
            overall_progress = (completed_items + current_item_contribution) / len(self.sub_items) * 100.0
            
            if message:
                current_item = self.sub_items[self.current_item_index] if self.current_item_index < len(self.sub_items) else "Unknown"
                full_message = f"{message} ({current_item}, {self.current_item_index + 1}/{len(self.sub_items)})"
            else:
                full_message = f"Processing {self.current_item_index + 1}/{len(self.sub_items)}"
            
            self.parent_tracker.update_phase_progress(overall_progress, full_message)
    
    def next_item(self):
        """Move to the next sub-item."""
        if self.current_item_index < len(self.sub_items) - 1:
            self.current_item_index += 1
            self.current_item_progress = 0.0
            self.update_item_progress(0.0)
    
    def complete_item(self):
        """Mark current item as complete and move to next."""
        self.current_item_progress = 100.0
        self.update_item_progress(100.0)
        self.next_item()


def create_console_progress_callback(use_progressbar: bool = True) -> Callable:
    """Create a console progress callback function with colors and improved formatting."""
    
    # ANSI color codes
    class Colors:
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        BOLD = '\033[1m'
        DIM = '\033[2m'
        RESET = '\033[0m'
    
    def console_callback(progress_data: Dict):
        current_phase = progress_data["current_phase"]
        overall_progress = progress_data["overall_progress"]
        eta = progress_data["estimated_time_remaining"]
        
        if current_phase:
            phase_name = current_phase.name if hasattr(current_phase, 'name') else current_phase["name"]
            phase_desc = current_phase.description if hasattr(current_phase, 'description') else current_phase["description"]
            
            # Color progress percentage based on completion
            if overall_progress < 25:
                progress_color = Colors.RED
            elif overall_progress < 50:
                progress_color = Colors.YELLOW
            elif overall_progress < 75:
                progress_color = Colors.CYAN
            else:
                progress_color = Colors.GREEN
            
            # Format ETA
            eta_str = ""
            if eta:
                if eta < 60:
                    eta_str = f" {Colors.DIM}(ETA: {eta:.0f}s){Colors.RESET}"
                else:
                    minutes = int(eta // 60)
                    seconds = int(eta % 60)
                    eta_str = f" {Colors.DIM}(ETA: {minutes}m {seconds}s){Colors.RESET}"
            
            # Create progress bar visual
            bar_length = 20
            filled_length = int(bar_length * overall_progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Format the complete progress line
            progress_line = (
                f"\r{Colors.BOLD}[{progress_color}{overall_progress:5.1f}%{Colors.RESET}{Colors.BOLD}] "
                f"{Colors.CYAN}{bar}{Colors.RESET} "
                f"{Colors.WHITE}{phase_name}:{Colors.RESET} "
                f"{phase_desc}"
                f"{eta_str}"
            )
            
            print(progress_line, end="", flush=True)
    
    return console_callback