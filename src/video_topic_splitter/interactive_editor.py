#!/usr/bin/env python3
"""
Interactive editor for segment summaries.

This module provides an interactive command-line interface for editing
segment summaries after video analysis is complete. Users can review
and modify AI-generated summaries to improve accuracy or add additional
insights.
"""

import json
import os
import shutil
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SegmentSummaryEditor:
    """
    Interactive editor for video segment summaries.
    
    Provides a menu-driven interface for reviewing and editing segment
    summaries, with automatic backup and validation of changes.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize the segment summary editor.
        
        Args:
            project_path: Path to the project directory containing segments
        """
        self.project_path = project_path
        self.segments_dir = os.path.join(project_path, "topic_segments")
        self.backup_dir = os.path.join(project_path, "backup_summaries")
        self.segments_data = []
        self.changes_made = False
        
    def load_segments(self) -> bool:
        """
        Load all segment summaries from the project directory.
        
        Returns:
            True if segments were loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.segments_dir):
                print(f"Error: Segments directory not found: {self.segments_dir}")
                return False
            
            self.segments_data = []
            
            # Find all segment directories (both old and new naming conventions)
            for item in os.listdir(self.segments_dir):
                item_path = os.path.join(self.segments_dir, item)
                if os.path.isdir(item_path) and item.startswith("segment_"):
                    summary_path = os.path.join(item_path, "segment_summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                                summary_data['_segment_dir'] = item_path
                                summary_data['_summary_path'] = summary_path
                                self.segments_data.append(summary_data)
                        except Exception as e:
                            logger.warning(f"Could not load summary from {summary_path}: {e}")
            
            # Sort by segment number
            self.segments_data.sort(key=lambda x: x.get('segment_number', 0))
            
            if not self.segments_data:
                print("No segment summaries found.")
                return False
            
            print(f"Loaded {len(self.segments_data)} segment summaries.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading segments: {e}")
            return False
    
    def display_segment_menu(self) -> None:
        """Display the main segment selection menu."""
        print("\n" + "="*60)
        print("SEGMENT SUMMARY EDITOR")
        print("="*60)
        print(f"Project: {os.path.basename(self.project_path)}")
        print(f"Total segments: {len(self.segments_data)}")
        print("-"*60)
        
        for i, segment in enumerate(self.segments_data, 1):
            segment_num = segment.get('segment_number', i)
            topic = segment.get('topic', 'Unknown Topic')
            duration = segment.get('duration', 0)
            insights_count = len(segment.get('key_insights', []))
            tech_count = len(segment.get('technical_elements', []))
            
            # Check for merge information
            merge_info = segment.get('merge_info', {})
            segments_merged = merge_info.get('segments_merged', 1)
            merge_confidence = merge_info.get('merge_confidence', 1.0)
            
            print(f"{i:2d}. Segment {segment_num}: {topic}")
            if segments_merged > 1:
                print(f"    Duration: {duration:.1f}s | Insights: {insights_count} | Tech: {tech_count}")
                print(f"    Merged: {segments_merged} segments | Confidence: {merge_confidence:.2f}")
            else:
                print(f"    Duration: {duration:.1f}s | Insights: {insights_count} | Tech: {tech_count}")
        
        print("-"*60)
        print("Commands:")
        print("  [1-N] - Edit specific segment")
        print("  [s]   - Save all changes")
        print("  [q]   - Quit editor")
        print("  [r]   - Refresh segment list")
        print("="*60)
    
    def edit_segment_summary(self, segment_index: int) -> bool:
        """
        Edit a specific segment summary.
        
        Args:
            segment_index: Index of the segment to edit (0-based)
            
        Returns:
            True if changes were made, False otherwise
        """
        if segment_index < 0 or segment_index >= len(self.segments_data):
            print("Invalid segment selection.")
            return False
        
        segment = self.segments_data[segment_index]
        segment_num = segment.get('segment_number', segment_index + 1)
        topic = segment.get('topic', 'Unknown Topic')
        
        print(f"\n{'='*60}")
        print(f"EDITING SEGMENT {segment_num}: {topic}")
        print(f"{'='*60}")
        
        changes_made = False
        
        while True:
            print(f"\nCurrent Summary:")
            print(f"Topic: {segment.get('topic', 'N/A')}")
            print(f"Duration: {segment.get('duration', 0):.1f} seconds")
            print(f"Confidence: {segment.get('confidence_score', 0):.2f}")
            
            # Show merge information if available
            merge_info = segment.get('merge_info', {})
            segments_merged = merge_info.get('segments_merged', 1)
            merge_confidence = merge_info.get('merge_confidence', 1.0)
            
            if segments_merged > 1:
                print(f"\nMerge Information:")
                print(f"  Segments merged: {segments_merged}")
                print(f"  Merge confidence: {merge_confidence:.2f}")
                
                # Show original segments if available
                original_segments = segment.get('original_segments', [])
                if original_segments and len(original_segments) > 1:
                    print(f"  Original segments: {len(original_segments)} parts")
                    for i, orig_seg in enumerate(original_segments[:3], 1):  # Show first 3
                        orig_topic = orig_seg.get('topic', 'Unknown')
                        orig_start = orig_seg.get('start', 0)
                        orig_end = orig_seg.get('end', 0)
                        print(f"    {i}. {orig_topic} ({orig_start:.1f}s-{orig_end:.1f}s)")
                    if len(original_segments) > 3:
                        print(f"    ... and {len(original_segments) - 3} more")
            
            print(f"\nKey Insights ({len(segment.get('key_insights', []))}):")
            for i, insight in enumerate(segment.get('key_insights', []), 1):
                print(f"  {i}. {insight}")
            
            print(f"\nTechnical Elements ({len(segment.get('technical_elements', []))}):")
            for i, tech in enumerate(segment.get('technical_elements', []), 1):
                print(f"  {i}. {tech}")
            
            print(f"\nModalities Analyzed: {', '.join(segment.get('modalities_analyzed', []))}")
            
            print(f"\n{'-'*40}")
            print("Edit Options:")
            print("  [1] - Edit topic")
            print("  [2] - Edit key insights")
            print("  [3] - Edit technical elements")
            print("  [4] - Edit confidence score")
            print("  [s] - Save changes to this segment")
            print("  [b] - Back to main menu")
            print(f"{'-'*40}")
            
            choice = input("Select option: ").strip().lower()
            
            if choice == '1':
                if self._edit_topic(segment):
                    changes_made = True
            elif choice == '2':
                if self._edit_key_insights(segment):
                    changes_made = True
            elif choice == '3':
                if self._edit_technical_elements(segment):
                    changes_made = True
            elif choice == '4':
                if self._edit_confidence_score(segment):
                    changes_made = True
            elif choice == 's':
                if changes_made:
                    if self._save_segment(segment):
                        print("Segment saved successfully!")
                        self.changes_made = True
                        return True
                else:
                    print("No changes to save.")
            elif choice == 'b':
                if changes_made:
                    save_choice = input("Save changes before leaving? (y/N): ").strip().lower()
                    if save_choice == 'y':
                        if self._save_segment(segment):
                            print("Segment saved successfully!")
                            self.changes_made = True
                            return True
                return changes_made
            else:
                print("Invalid choice. Please try again.")
        
        return changes_made
    
    def _edit_topic(self, segment: Dict[str, Any]) -> bool:
        """Edit the topic of a segment."""
        current_topic = segment.get('topic', '')
        print(f"\nCurrent topic: {current_topic}")
        new_topic = input("Enter new topic (or press Enter to keep current): ").strip()
        
        if new_topic and new_topic != current_topic:
            segment['topic'] = new_topic
            print(f"Topic updated to: {new_topic}")
            return True
        
        return False
    
    def _edit_key_insights(self, segment: Dict[str, Any]) -> bool:
        """Edit the key insights of a segment."""
        insights = segment.get('key_insights', [])
        changes_made = False
        
        while True:
            print(f"\nCurrent Key Insights ({len(insights)}):")
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight}")
            
            print(f"\nOptions:")
            print("  [a] - Add new insight")
            print("  [1-N] - Edit insight by number")
            print("  [d1-dN] - Delete insight by number (e.g., d1)")
            print("  [done] - Finish editing insights")
            
            choice = input("Select option: ").strip().lower()
            
            if choice == 'a':
                new_insight = input("Enter new insight: ").strip()
                if new_insight:
                    insights.append(new_insight)
                    changes_made = True
                    print("Insight added.")
            elif choice == 'done':
                if changes_made:
                    segment['key_insights'] = insights
                break
            elif choice.startswith('d') and len(choice) > 1:
                try:
                    index = int(choice[1:]) - 1
                    if 0 <= index < len(insights):
                        removed = insights.pop(index)
                        changes_made = True
                        print(f"Deleted: {removed}")
                    else:
                        print("Invalid insight number.")
                except ValueError:
                    print("Invalid delete command.")
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(insights):
                        print(f"Current insight: {insights[index]}")
                        new_insight = input("Enter new text (or press Enter to keep): ").strip()
                        if new_insight:
                            insights[index] = new_insight
                            changes_made = True
                            print("Insight updated.")
                    else:
                        print("Invalid insight number.")
                except ValueError:
                    print("Invalid choice.")
        
        return changes_made
    
    def _edit_technical_elements(self, segment: Dict[str, Any]) -> bool:
        """Edit the technical elements of a segment."""
        tech_elements = segment.get('technical_elements', [])
        changes_made = False
        
        while True:
            print(f"\nCurrent Technical Elements ({len(tech_elements)}):")
            for i, tech in enumerate(tech_elements, 1):
                print(f"  {i}. {tech}")
            
            print(f"\nOptions:")
            print("  [a] - Add new technical element")
            print("  [1-N] - Edit element by number")
            print("  [d1-dN] - Delete element by number (e.g., d1)")
            print("  [done] - Finish editing technical elements")
            
            choice = input("Select option: ").strip().lower()
            
            if choice == 'a':
                new_tech = input("Enter new technical element: ").strip()
                if new_tech:
                    tech_elements.append(new_tech)
                    changes_made = True
                    print("Technical element added.")
            elif choice == 'done':
                if changes_made:
                    segment['technical_elements'] = tech_elements
                break
            elif choice.startswith('d') and len(choice) > 1:
                try:
                    index = int(choice[1:]) - 1
                    if 0 <= index < len(tech_elements):
                        removed = tech_elements.pop(index)
                        changes_made = True
                        print(f"Deleted: {removed}")
                    else:
                        print("Invalid element number.")
                except ValueError:
                    print("Invalid delete command.")
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(tech_elements):
                        print(f"Current element: {tech_elements[index]}")
                        new_tech = input("Enter new text (or press Enter to keep): ").strip()
                        if new_tech:
                            tech_elements[index] = new_tech
                            changes_made = True
                            print("Technical element updated.")
                    else:
                        print("Invalid element number.")
                except ValueError:
                    print("Invalid choice.")
        
        return changes_made
    
    def _edit_confidence_score(self, segment: Dict[str, Any]) -> bool:
        """Edit the confidence score of a segment."""
        current_score = segment.get('confidence_score', 0.0)
        print(f"\nCurrent confidence score: {current_score:.2f}")
        
        try:
            new_score_str = input("Enter new confidence score (0.0-1.0, or press Enter to keep): ").strip()
            if new_score_str:
                new_score = float(new_score_str)
                if 0.0 <= new_score <= 1.0:
                    segment['confidence_score'] = new_score
                    print(f"Confidence score updated to: {new_score:.2f}")
                    return True
                else:
                    print("Confidence score must be between 0.0 and 1.0")
            return False
        except ValueError:
            print("Invalid number format.")
            return False
    
    def _save_segment(self, segment: Dict[str, Any]) -> bool:
        """Save a segment summary to disk."""
        try:
            summary_path = segment['_summary_path']
            
            # Create backup if it doesn't exist
            if not os.path.exists(self.backup_dir):
                os.makedirs(self.backup_dir)
            
            backup_filename = f"segment_{segment.get('segment_number', 'unknown')}_backup.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            if not os.path.exists(backup_path):
                shutil.copy2(summary_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Remove internal paths from saved data
            save_data = {k: v for k, v in segment.items() if not k.startswith('_')}
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving segment: {e}")
            print(f"Error saving segment: {e}")
            return False
    
    def save_all_changes(self) -> bool:
        """Save all pending changes to disk."""
        if not self.changes_made:
            print("No changes to save.")
            return True
        
        try:
            # Regenerate consolidated results
            self._update_consolidated_results()
            print("All changes saved and consolidated results updated.")
            return True
        except Exception as e:
            logger.error(f"Error saving changes: {e}")
            print(f"Error saving changes: {e}")
            return False
    
    def _update_consolidated_results(self) -> None:
        """Update the consolidated results file with any changes."""
        try:
            # Look for existing consolidated results
            results_path = os.path.join(self.project_path, "transcript_analysis_results.json")
            
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Update the detailed results with our modified segments
                if 'detailed_results' in results:
                    for i, segment in enumerate(self.segments_data):
                        # Find corresponding detailed result
                        for detailed_result in results['detailed_results']:
                            if detailed_result.get('segment_info', {}).get('segment_number') == segment.get('segment_number'):
                                # Update the multimodal_summary section
                                if 'multimodal_summary' in detailed_result:
                                    detailed_result['multimodal_summary'].update({
                                        'key_insights': segment.get('key_insights', []),
                                        'technical_elements': segment.get('technical_elements', []),
                                        'confidence_score': segment.get('confidence_score', 0.0)
                                    })
                                break
                
                # Save updated results
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info("Updated consolidated results file")
        
        except Exception as e:
            logger.warning(f"Could not update consolidated results: {e}")
    
    def run_interactive_editor(self) -> bool:
        """
        Run the interactive segment editor.
        
        Returns:
            True if the editor ran successfully, False otherwise
        """
        try:
            if not self.load_segments():
                return False
            
            while True:
                self.display_segment_menu()
                choice = input("\nEnter your choice: ").strip().lower()
                
                if choice == 'q':
                    if self.changes_made:
                        save_choice = input("Save changes before quitting? (y/N): ").strip().lower()
                        if save_choice == 'y':
                            self.save_all_changes()
                    print("Editor closed.")
                    break
                elif choice == 's':
                    self.save_all_changes()
                elif choice == 'r':
                    self.load_segments()
                    print("Segment list refreshed.")
                else:
                    try:
                        segment_index = int(choice) - 1
                        if 0 <= segment_index < len(self.segments_data):
                            self.edit_segment_summary(segment_index)
                        else:
                            print("Invalid segment number.")
                    except ValueError:
                        print("Invalid choice. Please try again.")
            
            return True
            
        except KeyboardInterrupt:
            print("\nEditor interrupted by user.")
            return False
        except Exception as e:
            logger.error(f"Error in interactive editor: {e}")
            print(f"Error in interactive editor: {e}")
            return False


def launch_segment_editor(project_path: str) -> bool:
    """
    Launch the interactive segment summary editor.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        True if editor ran successfully, False otherwise
    """
    try:
        editor = SegmentSummaryEditor(project_path)
        return editor.run_interactive_editor()
    except Exception as e:
        logger.error(f"Failed to launch segment editor: {e}")
        return False