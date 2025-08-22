#!/usr/bin/env python3
"""
Prompt template utilities for loading and processing prompt templates.

This module provides utilities for loading prompt templates from files
and performing template variable substitution for dynamic prompt generation.
"""

import os
import logging
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


def load_prompt_template(template_path: str) -> str:
    """
    Load a prompt template from a file.
    
    Args:
        template_path: Path to the prompt template file
        
    Returns:
        The content of the prompt template file
        
    Raises:
        FileNotFoundError: If the template file does not exist
        IOError: If there's an error reading the file
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Successfully loaded prompt template from {template_path}")
        return content
        
    except IOError as e:
        error_msg = f"Error reading prompt template {template_path}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg)


def substitute_template_variables(template: str, variables: Dict[str, Any]) -> str:
    """
    Replace template placeholders with actual values.
    
    Supports placeholders in the format {{VARIABLE_NAME}}.
    
    Args:
        template: The template string with placeholders
        variables: Dictionary of variable names to replacement values
        
    Returns:
        Template with all placeholders replaced by their values
        
    Examples:
        >>> template = "Hello {{NAME}}, welcome to {{PROJECT}}!"
        >>> variables = {"NAME": "Alice", "PROJECT": "MyApp"}
        >>> substitute_template_variables(template, variables)
        "Hello Alice, welcome to MyApp!"
    """
    if not variables:
        return template
    
    result = template
    
    # Find all placeholders in the format {{VARIABLE_NAME}}
    placeholders = re.findall(r'\{\{([^}]+)\}\}', template)
    
    for placeholder in placeholders:
        placeholder_key = placeholder.strip()
        
        if placeholder_key in variables:
            # Convert value to string and replace placeholder
            value = str(variables[placeholder_key])
            pattern = r'\{\{\s*' + re.escape(placeholder_key) + r'\s*\}\}'
            result = re.sub(pattern, value, result)
            logger.debug(f"Replaced {{{{ {placeholder_key} }}}} with: {value[:100]}...")
        else:
            logger.warning(f"Template variable '{placeholder_key}' not found in provided variables")
    
    return result


def load_and_process_template(template_path: str, variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Load a prompt template and substitute variables in one operation.
    
    This is a convenience function that combines template loading and variable
    substitution into a single call.
    
    Args:
        template_path: Path to the prompt template file
        variables: Optional dictionary of template variables to substitute
        
    Returns:
        Processed template with variables substituted
        
    Raises:
        FileNotFoundError: If the template file does not exist
        IOError: If there's an error reading the file
    """
    template = load_prompt_template(template_path)
    
    if variables:
        template = substitute_template_variables(template, variables)
    
    return template


def get_default_template_path(template_name: str, prompts_dir: Optional[str] = None) -> str:
    """
    Get the default path for a prompt template.
    
    Args:
        template_name: Name of the template file (with or without .md extension)
        prompts_dir: Optional custom prompts directory path
        
    Returns:
        Full path to the template file
    """
    if prompts_dir is None:
        # Default to prompts directory relative to this module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(os.path.dirname(current_dir), "prompts")
    
    # Add .md extension if not present
    if not template_name.endswith('.md'):
        template_name += '.md'
    
    return os.path.join(prompts_dir, template_name)


def validate_template_variables(template: str, required_variables: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate that a template contains the expected variables.
    
    Args:
        template: The template string to validate
        required_variables: Optional list of required variable names
        
    Returns:
        Dictionary with validation results:
        - 'found_variables': List of variables found in template
        - 'missing_variables': List of required variables not found
        - 'is_valid': Boolean indicating if all required variables are present
    """
    # Find all placeholders in the template
    found_variables = re.findall(r'\{\{([^}]+)\}\}', template)
    found_variables = [var.strip() for var in found_variables]
    
    validation_result = {
        'found_variables': list(set(found_variables)),
        'missing_variables': [],
        'is_valid': True
    }
    
    if required_variables:
        missing_variables = [var for var in required_variables if var not in found_variables]
        validation_result['missing_variables'] = missing_variables
        validation_result['is_valid'] = len(missing_variables) == 0
    
    return validation_result