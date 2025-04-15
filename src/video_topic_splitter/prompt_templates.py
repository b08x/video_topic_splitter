#!/usr/bin/env python3
"""Register-aware prompt templates for different technical contexts.

This module provides a collection of prompt templates tailored for analyzing
video content within specific technical domains (registers), such as IT workflows,
generative AI, and technical support. It offers structured prompts designed
to guide large language models (LLMs) in extracting relevant information
and identifying patterns specific to each domain.

The module includes:
- A `RegisterTemplates` class containing static methods to generate prompts
  for topic identification and detailed analysis within each register.
- Helper functions (`get_topic_prompt`, `get_analysis_prompt`, `get_visual_topic_prompt`) 
  to dynamically select the appropriate prompt based on a specified register string.
"""

from typing import Dict, Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)


class RegisterTemplates:
    """Manages and provides prompt templates for different technical registers.

    This class acts as a namespace for static methods, each generating a
    specific prompt template tailored to a technical domain (register).
    These templates are designed to be used with large language models (LLMs)
    for tasks like topic identification or detailed analysis of video transcripts
    or segments.

    The prompts guide the LLM to focus on domain-specific elements, such as
    commands, configurations, model parameters, error messages, or resolution
    steps, depending on the chosen register (e.g., IT workflow, generative AI,
    technical support).
    """

    @staticmethod
    def get_it_workflow_topic_prompt(context: str) -> str:
        """Generate a prompt for identifying the main topic in an IT workflow context.

        This prompt instructs an LLM to analyze a given text segment (`context`)
        with a focus on identifying patterns typical of IT workflows. It asks
        the LLM to consider technical procedures, system commands, software
        configurations, system interactions, technical jargon, and step-by-step
        processes.

        The expected output format is a JSON object containing the main topic,
        relevant keywords, the relationship of the segment to the previous one
        (CONTINUATION, SHIFT, NEW), and a confidence score.

        Args:
            context: The text segment (e.g., from a transcript) to be analyzed.

        Returns:
            A formatted string containing the prompt for IT workflow topic analysis.
        """
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
        """Generate a prompt for identifying the main topic in a generative AI context.

        This prompt instructs an LLM to analyze a given text segment (`context`)
        with a focus on identifying patterns related to generative AI. It asks
        the LLM to consider AI model architectures, prompt engineering, model
        output patterns, implementation strategies, and API integration.

        The expected output format is a JSON object containing the main AI topic,
        relevant keywords (model terms, parameters), the relationship to the
        previous segment, and a confidence score.

        Args:
            context: The text segment (e.g., from a transcript) to be analyzed.

        Returns:
            A formatted string containing the prompt for generative AI topic analysis.
        """
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
        """Generate a prompt for identifying the main topic in a technical support context.

        This prompt instructs an LLM to analyze a given text segment (`context`)
        with a focus on identifying patterns typical of technical support scenarios.
        It asks the LLM to consider problem descriptions, diagnostic procedures,
        error patterns, resolution steps, and verification methods.

        The expected output format is a JSON object containing the main support topic,
        relevant keywords (error terms, solutions), the relationship to the
        previous segment, and a confidence score.

        Args:
            context: The text segment (e.g., from a transcript) to be analyzed.

        Returns:
            A formatted string containing the prompt for technical support topic analysis.
        """
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
    def get_educational_topic_prompt(context: str) -> str:
        """Generate a prompt for identifying the main topic in an educational context.

        This prompt instructs an LLM to analyze a given text segment (`context`)
        with a focus on identifying patterns typical of educational content.
        It asks the LLM to consider learning objectives, key concepts,
        instructional methods, examples, and assessment approaches.

        The expected output format is a JSON object containing the main educational topic,
        relevant keywords, the relationship to the previous segment, and a confidence score.

        Args:
            context: The text segment (e.g., from a transcript) to be analyzed.

        Returns:
            A formatted string containing the prompt for educational topic analysis.
        """
        return f"""
        Analyze this segment with a focus on educational content patterns:

        {context}

        Consider:
        1. Learning objectives and outcomes
        2. Key concepts and principles
        3. Instructional methods and approaches
        4. Examples and illustrations
        5. Assessment and practice elements

        Identify:
        - Main educational topic
        - Key concepts
        - Teaching patterns
        - Learning activities

        Format response as JSON:
        {{
            "topic": "main educational topic",
            "keywords": ["concept 1", "principle 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85
        }}
        """

    @staticmethod
    def get_it_workflow_analysis_prompt(context: str, transcript: str) -> str:
        """Generate a detailed analysis prompt for an IT workflow video segment.

        This prompt is designed for a more in-depth analysis of a video segment's
        transcript, specifically focusing on IT workflow elements. It asks the LLM
        to identify software tools, command-line operations, configuration steps,
        technical procedures, and integration patterns. It emphasizes capturing
        specific technical details, commands, and sequences.

        Note: The 'context' parameter is currently included for potential future use
        or consistency but is not directly used in the f-string formatting of this
        specific prompt template.

        Args:
            context: Additional context about the segment (currently unused in prompt).
            transcript: The transcript text of the video segment to be analyzed.

        Returns:
            A formatted string containing the detailed analysis prompt for IT workflows.
        """
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
        """Generate a detailed analysis prompt for a generative AI video segment.

        This prompt guides an LLM to perform a detailed analysis of a video segment's
        transcript, focusing on generative AI concepts. It asks for identification
        of AI models, prompt techniques, parameters, implementation strategies, and
        API integration. Emphasis is placed on model-specific details, parameter
        adjustments, output characteristics, and performance considerations.

        Note: The 'context' parameter is currently included for potential future use
        or consistency but is not directly used in the f-string formatting of this
        specific prompt template.

        Args:
            context: Additional context about the segment (currently unused in prompt).
            transcript: The transcript text of the video segment to be analyzed.

        Returns:
            A formatted string containing the detailed analysis prompt for generative AI.
        """
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
        """Generate a detailed analysis prompt for a technical support video segment.

        This prompt directs an LLM to conduct a detailed analysis of a video segment's
        transcript within a technical support context. It requests identification of
        problem descriptions, error messages, diagnostic steps, resolution procedures,
        and verification methods. Special attention is requested for error patterns,
        diagnostic sequences, resolution details, and system state changes.

        Note: The 'context' parameter is currently included for potential future use
        or consistency but is not directly used in the f-string formatting of this
        specific prompt template.

        Args:
            context: Additional context about the segment (currently unused in prompt).
            transcript: The transcript text of the video segment to be analyzed.

        Returns:
            A formatted string containing the detailed analysis prompt for technical support.
        """
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

    @staticmethod
    def get_educational_analysis_prompt(context: str, transcript: str) -> str:
        """Generate a detailed analysis prompt for an educational video segment.

        This prompt guides an LLM to perform a detailed analysis of a video segment's
        transcript, focusing on educational content. It asks for identification
        of learning objectives, key concepts, instructional methods, examples,
        and assessment approaches. Emphasis is placed on pedagogical patterns,
        concept explanations, and learning activities.

        Args:
            context: Additional context about the segment (currently unused in prompt).
            transcript: The transcript text of the video segment to be analyzed.

        Returns:
            A formatted string containing the detailed analysis prompt for educational content.
        """
        return f"""
        Analyze this video segment focusing on educational content patterns.

        Transcript: '{transcript}'

        Please identify:
        1. Learning objectives and outcomes
        2. Key concepts and principles being taught
        3. Instructional methods and approaches
        4. Examples and illustrations used
        5. Assessment and practice elements

        Pay special attention to:
        - Pedagogical patterns
        - Concept explanations
        - Teaching techniques
        - Learning activities
        - Knowledge assessment

        Format the findings with clear educational details and teaching patterns.
        """

    # New methods for visual topic analysis
    @staticmethod
    def get_it_workflow_visual_topic_prompt(context: str) -> str:
        """Generate a prompt for identifying topics in IT workflow content with visual context.

        This prompt instructs an LLM to analyze both textual and visual elements
        of a video segment in an IT workflow context. It asks the LLM to consider
        technical procedures, system commands, software interfaces, visual demonstrations,
        and technical environments shown on screen.

        The expected output format is a JSON object containing both textual and visual
        topic information, keywords, relationships, and confidence scores.

        Args:
            context: The combined text and visual descriptions to be analyzed.

        Returns:
            A formatted string containing the prompt for IT workflow visual topic analysis.
        """
        return f"""
        Analyze this segment with a focus on IT workflow patterns, considering both text and visual elements:

        {context}

        Consider in the TEXT:
        1. Technical procedures and system commands
        2. Software configuration steps
        3. System interaction patterns
        4. Technical terminology and jargon
        5. Step-by-step process structures

        Consider in the VISUALS:
        1. Software interfaces and tools shown
        2. Command-line environments
        3. Configuration screens
        4. Visual demonstrations of procedures
        5. Technical environments and setups

        Format response as JSON:
        {{
            "topic": "main workflow topic from text",
            "keywords": ["technical term 1", "command 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85,
            "visual_topic": "what is being shown visually",
            "visual_keywords": ["interface element 1", "visual cue 2", ...],
            "visual_relationship": "CONTINUATION|SHIFT|NEW",
            "visual_summary": "Brief description of what's being shown on screen"
        }}
        """

    @staticmethod
    def get_gen_ai_visual_topic_prompt(context: str) -> str:
        """Generate a prompt for identifying topics in generative AI content with visual context.

        This prompt instructs an LLM to analyze both textual and visual elements
        of a video segment in a generative AI context. It asks the LLM to consider
        AI model architectures, prompt engineering, model outputs, visual demonstrations
        of AI capabilities, and UI interfaces for AI tools.

        The expected output format is a JSON object containing both textual and visual
        topic information, keywords, relationships, and confidence scores.

        Args:
            context: The combined text and visual descriptions to be analyzed.

        Returns:
            A formatted string containing the prompt for generative AI visual topic analysis.
        """
        return f"""
        Analyze this segment with a focus on generative AI patterns, considering both text and visual elements:

        {context}

        Consider in the TEXT:
        1. AI model architectures and parameters
        2. Prompt engineering techniques
        3. Model output patterns
        4. Implementation strategies
        5. API integration methods

        Consider in the VISUALS:
        1. AI interfaces and dashboards
        2. Visual demonstrations of AI capabilities
        3. Model output examples
        4. Prompt construction interfaces
        5. Visual representations of AI concepts

        Format response as JSON:
        {{
            "topic": "main AI topic from text",
            "keywords": ["model term 1", "parameter 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85,
            "visual_topic": "what is being shown visually",
            "visual_keywords": ["interface element 1", "output example 2", ...],
            "visual_relationship": "CONTINUATION|SHIFT|NEW",
            "visual_summary": "Brief description of what's being shown on screen"
        }}
        """

    @staticmethod
    def get_tech_support_visual_topic_prompt(context: str) -> str:
        """Generate a prompt for identifying topics in tech support content with visual context.

        This prompt instructs an LLM to analyze both textual and visual elements
        of a video segment in a technical support context. It asks the LLM to consider
        problem descriptions, diagnostic procedures, error messages, visual demonstrations
        of issues, and interface elements showing errors or solutions.

        The expected output format is a JSON object containing both textual and visual
        topic information, keywords, relationships, and confidence scores.

        Args:
            context: The combined text and visual descriptions to be analyzed.

        Returns:
            A formatted string containing the prompt for technical support visual topic analysis.
        """
        return f"""
        Analyze this segment with a focus on technical support patterns, considering both text and visual elements:

        {context}

        Consider in the TEXT:
        1. Problem descriptions and symptoms
        2. Diagnostic procedures
        3. Error patterns and messages
        4. Resolution steps
        5. Verification methods

        Consider in the VISUALS:
        1. Error screens and messages
        2. Diagnostic tool interfaces
        3. Visual demonstrations of issues
        4. Step-by-step resolution visuals
        5. System state indicators

        Format response as JSON:
        {{
            "topic": "main support topic from text",
            "keywords": ["error term 1", "solution 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85,
            "visual_topic": "what is being shown visually",
            "visual_keywords": ["error screen 1", "interface element 2", ...],
            "visual_relationship": "CONTINUATION|SHIFT|NEW",
            "visual_summary": "Brief description of what's being shown on screen"
        }}
        """

    @staticmethod
    def get_educational_visual_topic_prompt(context: str) -> str:
        """Generate a prompt for identifying topics in educational content with visual context.

        This prompt instructs an LLM to analyze both textual and visual elements
        of a video segment in an educational context. It asks the LLM to consider
        learning objectives, key concepts, instructional methods, visual aids,
        diagrams, demonstrations, and educational interfaces.

        The expected output format is a JSON object containing both textual and visual
        topic information, keywords, relationships, and confidence scores.

        Args:
            context: The combined text and visual descriptions to be analyzed.

        Returns:
            A formatted string containing the prompt for educational visual topic analysis.
        """
        return f"""
        Analyze this segment with a focus on educational content patterns, considering both text and visual elements:

        {context}

        Consider in the TEXT:
        1. Learning objectives and outcomes
        2. Key concepts and principles
        3. Instructional methods and approaches
        4. Examples and illustrations
        5. Assessment and practice elements

        Consider in the VISUALS:
        1. Visual aids and diagrams
        2. Demonstrations and examples
        3. Educational interfaces
        4. Visual representations of concepts
        5. Learning activities shown on screen

        Format response as JSON:
        {{
            "topic": "main educational topic from text",
            "keywords": ["concept 1", "principle 2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85,
            "visual_topic": "what is being shown visually",
            "visual_keywords": ["diagram 1", "visual example 2", ...],
            "visual_relationship": "CONTINUATION|SHIFT|NEW",
            "visual_summary": "Brief description of what's being shown on screen"
        }}
        """


def get_topic_prompt(register: str, context: str) -> str:
    """Retrieves the appropriate topic identification prompt for a given technical register.

    Selects and formats a topic analysis prompt based on the provided `register`
    string. It maps register names (e.g., "it-workflow", "gen-ai") to the
    corresponding static methods in the `RegisterTemplates` class.

    If the specified register is not found in the predefined mapping, it defaults
    to using the IT workflow topic prompt.

    Args:
        register: A string identifying the technical register (e.g., "it-workflow",
                  "gen-ai", "tech-support").
        context: The text segment to be included in the prompt for analysis.

    Returns:
        A formatted string containing the selected topic analysis prompt.
    """
    templates = {
        "it-workflow": RegisterTemplates.get_it_workflow_topic_prompt,
        "gen-ai": RegisterTemplates.get_gen_ai_topic_prompt,
        "tech-support": RegisterTemplates.get_tech_support_topic_prompt,
        "educational": RegisterTemplates.get_educational_topic_prompt,
    }
    # Get the appropriate function from the dictionary, defaulting if not found.
    # Then call the retrieved function with the context.
    prompt_func = templates.get(register, RegisterTemplates.get_it_workflow_topic_prompt)
    return prompt_func(context)


def get_analysis_prompt(register: str, context: str, transcript: str) -> str:
    """Retrieves the appropriate detailed analysis prompt for a given technical register.

    Selects and formats a detailed analysis prompt based on the provided `register`
    string. It maps register names (e.g., "it-workflow", "gen-ai") to the
    corresponding static methods in the `RegisterTemplates` class that generate
    prompts for analyzing full transcripts.

    If the specified register is not found in the predefined mapping, it defaults
    to using the IT workflow analysis prompt.

    Args:
        register: A string identifying the technical register (e.g., "it-workflow",
                  "gen-ai", "tech-support").
        context: Additional context for the analysis (passed to the template function).
        transcript: The transcript text to be included in the prompt for analysis.

    Returns:
        A formatted string containing the selected detailed analysis prompt.
    """
    templates = {
        "it-workflow": RegisterTemplates.get_it_workflow_analysis_prompt,
        "gen-ai": RegisterTemplates.get_gen_ai_analysis_prompt,
        "tech-support": RegisterTemplates.get_tech_support_analysis_prompt,
        "educational": RegisterTemplates.get_educational_analysis_prompt,
    }
    # Get the appropriate function from the dictionary, defaulting if not found.
    # Then call the retrieved function with context and transcript.
    prompt_func = templates.get(register, RegisterTemplates.get_it_workflow_analysis_prompt)
    return prompt_func(context, transcript)


def get_visual_topic_prompt(register: str, context: str) -> str:
    """Retrieves the appropriate visual topic analysis prompt for a given technical register.

    Selects and formats a visual topic analysis prompt based on the provided `register`
    string. It maps register names (e.g., "it-workflow", "gen-ai") to the
    corresponding static methods in the `RegisterTemplates` class that generate
    prompts for analyzing both text and visual elements.

    If the specified register is not found in the predefined mapping, it defaults
    to using the IT workflow visual topic prompt.

    Args:
        register: A string identifying the technical register (e.g., "it-workflow",
                  "gen-ai", "tech-support", "educational").
        context: The combined text and visual descriptions to be included in the prompt.

    Returns:
        A formatted string containing the selected visual topic analysis prompt.
    """
    templates = {
        "it-workflow": RegisterTemplates.get_it_workflow_visual_topic_prompt,
        "gen-ai": RegisterTemplates.get_gen_ai_visual_topic_prompt,
        "tech-support": RegisterTemplates.get_tech_support_visual_topic_prompt,
        "educational": RegisterTemplates.get_educational_visual_topic_prompt,
    }
    # Get the appropriate function from the dictionary, defaulting if not found.
    # Then call the retrieved function with the context.
    prompt_func = templates.get(register, RegisterTemplates.get_it_workflow_visual_topic_prompt)
    logger.debug(f"Using visual topic prompt for register: {register}")
    return prompt_func(context)