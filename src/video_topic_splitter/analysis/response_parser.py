"""Response parsing and validation for topic analysis."""

import json
import re
from typing import Dict, List, Optional, Any


class TopicAnalysisError(Exception):
    """Base exception for topic analysis errors."""
    pass


class APIResponseError(TopicAnalysisError):
    """Exception raised when API response cannot be parsed."""
    pass


class ResponseParser:
    """Handles parsing and validation of API responses."""
    
    @staticmethod
    def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from API content with multiple fallback strategies."""
        content = content.strip()
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        patterns = [
            r"```json\s*({.*?})\s*```",
            r"```\s*({.*?})\s*```",
            r"```json\s*\n({.*?})\n```",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON object in mixed content
        json_pattern = r'({\s*"[^"]+"\s*:[^}]+})'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Extract individual fields using regex
        topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', content)
        if topic_match:
            result = {
                "topic": topic_match.group(1),
                "keywords": [],
                "relationship": "NEW",
                "confidence": 50
            }
            
            # Extract keywords
            keywords_match = re.search(r'"keywords"\s*:\s*\[([^\]]+)\]', content)
            if keywords_match:
                keywords_str = keywords_match.group(1)
                keywords = re.findall(r'"([^"]+)"', keywords_str)
                result["keywords"] = keywords
            
            # Extract relationship
            relationship_match = re.search(r'"relationship"\s*:\s*"([^"]+)"', content)
            if relationship_match:
                result["relationship"] = relationship_match.group(1)
            
            # Extract confidence
            confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', content)
            if confidence_match:
                result["confidence"] = int(confidence_match.group(1))
            
            return result
        
        return None
    
    @staticmethod
    def validate_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parsed response."""
        validated = {
            "topic": response.get("topic", "Uncategorized"),
            "keywords": response.get("keywords", []),
            "relationship": response.get("relationship", "NEW"),
            "confidence": response.get("confidence", 50)
        }
        
        # Validate topic
        if not validated["topic"] or not isinstance(validated["topic"], str):
            validated["topic"] = "Uncategorized"
        
        # Validate keywords
        if not isinstance(validated["keywords"], list):
            validated["keywords"] = []
        validated["keywords"] = [str(kw) for kw in validated["keywords"] if kw]
        
        # Validate relationship
        valid_relationships = ["CONTINUATION", "SHIFT", "NEW"]
        if validated["relationship"] not in valid_relationships:
            validated["relationship"] = "NEW"
        
        # Validate confidence
        try:
            confidence = int(validated["confidence"])
            validated["confidence"] = max(0, min(100, confidence))
        except (ValueError, TypeError):
            validated["confidence"] = 50
        
        return validated