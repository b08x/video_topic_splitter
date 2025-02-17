# api/gemini.py
#!/usr/bin/env python3
"""Gemini API integration for video analysis."""

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_with_gemini(prompt, image=None):
    """Analyze content using Google's Gemini API."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    if image:
        response = model.generate_content([prompt, image])
    else:
        response = model.generate_content(prompt)
    return response.text
