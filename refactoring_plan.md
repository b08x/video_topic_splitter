# Refactoring Plan for Circular Import Fix

## Problem
There is a circular import between visual_analysis.py and frame_analysis.py:
- visual_analysis.py imports ContextualFrameAnalyzer from frame_analysis.py
- frame_analysis.py imports detect_software_logos from visual_analysis.py

This creates a circular dependency where each module needs the other to be fully initialized first.

## Solution
Create a new module for software detection functionality that both can import independently.

## Implementation Details

### 1. Create New Module Structure
```
src/video_topic_splitter/processing/software/
├── __init__.py
└── software_detection.py
```

### 2. Software Detection Module Implementation
File: `software_detection.py`
- Move detect_software_logos function from visual_analysis.py
- Required imports: cv2, logging, numpy, os
- Keep exact same function signature and implementation
- Add proper docstrings and logging

### 3. Update Existing Files

#### visual_analysis.py changes:
- Remove detect_software_logos function
- Add import: `from ..processing.software.software_detection import detect_software_logos`
- Keep all other functionality intact
- No changes needed to function calls since signature remains same

#### frame_analysis.py changes:
- Update import to: `from ..processing.software.software_detection import detect_software_logos`
- Remove old import from visual_analysis
- No changes needed to function calls

### 4. Testing Steps
1. Verify imports resolve correctly
2. Test software logo detection functionality:
   - Test with existing logo database
   - Verify detection results match previous implementation
   - Check logging works correctly
3. Run full video analysis pipeline to ensure no regressions

## Expected Results
- Clean separation of software detection logic
- No more circular imports
- Improved code organization
- No changes to functionality or output

## Next Steps
1. Switch to code mode for implementation
2. Create software package structure
3. Move and update code
4. Test changes