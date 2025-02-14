#!/usr/bin/env python3
"""Register-aware prompt templates for different technical contexts."""

from typing import Dict, Optional


class RegisterTemplates:
    """Manages prompt templates for different technical registers."""

    @staticmethod
    def get_it_workflow_topic_prompt(context: str) -> str:
        """Generate prompt for IT workflow analysis."""
        return f"""
        Analyze this segment with a focus on IT workflow patterns:

        {context}

        Consider:
        1. Technical procedures and system commands
        2. Software configuration steps
        3. System interaction patterns
        4. Technical terminology and jargon
        5. Step-by-step process structures

        Identify:
        - Main workflow topic
        - Technical tools and commands used
        - Configuration patterns
        - System interaction sequences

        Format response as JSON:
        {{
            "topic": "main workflow topic",
            "keywords": ["technical term 1", "command 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85
        }}
        """

    @staticmethod
    def get_gen_ai_topic_prompt(context: str) -> str:
        """Generate prompt for generative AI analysis."""
        return f"""
        Analyze this segment with a focus on generative AI patterns:

        {context}

        Consider:
        1. AI model architectures and parameters
        2. Prompt engineering techniques
        3. Model output patterns
        4. Implementation strategies
        5. API integration methods

        Identify:
        - Main AI topic
        - Model-specific terminology
        - Technical parameters
        - Implementation patterns

        Format response as JSON:
        {{
            "topic": "main AI topic",
            "keywords": ["model term 1", "parameter 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85
        }}
        """

    @staticmethod
    def get_tech_support_topic_prompt(context: str) -> str:
        """Generate prompt for technical support analysis."""
        return f"""
        Analyze this segment with a focus on technical support patterns:

        {context}

        Consider:
        1. Problem descriptions and symptoms
        2. Diagnostic procedures
        3. Error patterns and messages
        4. Resolution steps
        5. Verification methods

        Identify:
        - Main support topic
        - Technical issues
        - Resolution patterns
        - Verification steps

        Format response as JSON:
        {{
            "topic": "main support topic",
            "keywords": ["error term 1", "solution 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85
        }}
        """

    @staticmethod
    def get_it_workflow_analysis_prompt(context: str, transcript: str) -> str:
        """Generate Gemini prompt for IT workflow video analysis."""
        return f"""
        Analyze this video segment focusing on IT workflow patterns.

        Transcript: '{transcript}'

        Please identify:
        1. Software tools and applications in use
        2. Command-line operations and syntax
        3. System configuration steps
        4. Technical procedures and workflows
        5. Integration patterns between tools

        Pay special attention to:
        - Technical terminology and commands
        - Tool-specific operations
        - System interaction patterns
        - Configuration sequences
        - Workflow transitions

        Format the findings with clear technical details and command syntax.
        """

    @staticmethod
    def get_gen_ai_analysis_prompt(context: str, transcript: str) -> str:
        """Generate Gemini prompt for generative AI video analysis."""
        return f"""
        Analyze this video segment focusing on generative AI patterns.

        Transcript: '{transcript}'

        Please identify:
        1. AI models and architectures discussed
        2. Prompt engineering techniques
        3. Model parameters and configurations
        4. Implementation strategies
        5. API integration patterns

        Pay special attention to:
        - Model-specific terminology
        - Parameter adjustments
        - Output patterns
        - Integration methods
        - Performance considerations

        Format the findings with clear technical details and implementation patterns.
        """

    @staticmethod
    def get_tech_support_analysis_prompt(context: str, transcript: str) -> str:
        """Generate Gemini prompt for technical support video analysis."""
        return f"""
        Analyze this video segment focusing on technical support patterns.

        Transcript: '{transcript}'

        Please identify:
        1. Problem descriptions and symptoms
        2. Error messages and patterns
        3. Diagnostic procedures
        4. Resolution steps
        5. Verification methods

        Pay special attention to:
        - Error patterns and messages
        - Diagnostic sequences
        - Resolution procedures
        - Verification steps
        - System state changes

        Format the findings with clear technical details and resolution patterns.
        """


def get_topic_prompt(register: str, context: str) -> str:
    """Get the appropriate topic analysis prompt for the given register."""
    templates = {
        "it-workflow": RegisterTemplates.get_it_workflow_topic_prompt,
        "gen-ai": RegisterTemplates.get_gen_ai_topic_prompt,
        "tech-support": RegisterTemplates.get_tech_support_topic_prompt,
    }
    return templates.get(register, RegisterTemplates.get_it_workflow_topic_prompt)(
        context
    )


def get_analysis_prompt(register: str, context: str, transcript: str) -> str:
    """Get the appropriate video analysis prompt for the given register."""
    templates = {
        "it-workflow": RegisterTemplates.get_it_workflow_analysis_prompt,
        "gen-ai": RegisterTemplates.get_gen_ai_analysis_prompt,
        "tech-support": RegisterTemplates.get_tech_support_analysis_prompt,
    }
    return templates.get(register, RegisterTemplates.get_it_workflow_analysis_prompt)(
        context, transcript
    )
