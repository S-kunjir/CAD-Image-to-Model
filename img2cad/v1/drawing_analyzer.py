"""
Universal CAD drawing analyzer that works for any 2D CAD image
"""
import re
import json
from typing import Dict
from loguru import logger

from ..chat_models import ChatModelParameters
from ..image import ImageData


class DrawingAnalyzer:
    """Analyze any 2D CAD drawing and extract structured information"""
    
    def __init__(self, model_type: str = "gpt"):
        self.model_type = model_type
        self.llm = ChatModelParameters.from_model_name(model_type).create_chat_model()
    
    def analyze(self, image_data: ImageData) -> Dict:
        """Analyze CAD drawing and return structured analysis"""
        
        analysis_prompt = """You are an expert mechanical engineer analyzing 2D CAD drawings.

TASK: Analyze this engineering drawing COMPLETELY and extract ALL information needed to recreate it in 3D.

ANALYSIS FORMAT - Provide in JSON format:
{
    "drawing_type": "orthographic|isometric|section|assembly|part",
    "views": ["front", "top", "side", "isometric"],
    "dimensions": {
        "overall_length": "100 mm",
        "hole_diameter": "10 mm",
        ...
    },
    "features": ["holes", "slots", "fillets", "chamfers", "threads", "patterns"],
    "complexity": "simple|moderate|complex",
    "is_assembly": true|false,
    "construction_strategy": "Start with base shape, then add holes, then fillets",
    "critical_dimensions": ["overall_length", "hole_spacing"],
    "symmetry": "x-axis|y-axis|radial|none",
    "material_thickness": "5 mm",
    "tolerances": "±0.1 mm",
    "notes": "Any additional important information"
}

IMPORTANT:
1. Extract ALL visible dimensions
2. Identify ALL geometric features
3. Determine if it's an assembly or single part
4. Note symmetry and patterns
5. Consider manufacturing constraints
6. Think about how to build it in 3D step by step

Provide ONLY valid JSON output."""
        
        try:
            # SIMPLE FIX: Use direct approach without complex template
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            # Create the image URL
            image_url = f"data:image/{image_data.type};base64,{image_data.data}"
            
            # Create message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
            
            # Get response directly
            response = self.llm.invoke([message])
            text_result = response.content
            
            logger.info(f"Got analysis response: {text_result[:200]}...")
            return self._parse_analysis(text_result)
            
        except Exception as e:
            logger.error(f"Error in drawing analysis: {str(e)[:200]}")
            # Return default analysis on error
            return self._get_default_analysis()
    
    def _parse_analysis(self, text: str) -> Dict:
        """Parse LLM response into structured analysis"""
        try:
            # Clean the text
            text = text.strip()
            logger.info(f"Parsing analysis text (first 500 chars): {text[:500]}")
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                analysis = json.loads(json_str)
                logger.info("✅ Successfully parsed JSON analysis")
            else:
                # Fallback to manual parsing
                logger.warning("⚠️ Could not find JSON, using manual parsing")
                analysis = self._manual_parse(text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, using manual parsing")
            analysis = self._manual_parse(text)
        except Exception as e:
            logger.error(f"Analysis parsing error: {e}")
            analysis = self._get_default_analysis()
        
        # Ensure required fields with defaults
        defaults = {
            "drawing_type": "orthographic",
            "views": ["front", "top", "side"],
            "dimensions": {},
            "features": ["extrude"],
            "complexity": "moderate",
            "is_assembly": False,
            "construction_strategy": "Start with base shape from main view, extrude to thickness, then add features",
            "critical_dimensions": [],
            "symmetry": "none",
            "material_thickness": "",
            "tolerances": "",
            "notes": "",
            "raw_text": text[:500]
        }
        
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
            elif analysis[key] is None:
                analysis[key] = default_value
        
        return analysis
    
    def _manual_parse(self, text: str) -> Dict:
        """Manual parsing if JSON fails"""
        logger.info("Using manual parsing for analysis")
        
        analysis = {
            "drawing_type": "orthographic",
            "views": ["front", "top", "side"],
            "dimensions": {},
            "features": [],
            "complexity": "moderate",
            "is_assembly": False,
            "construction_strategy": "",
            "critical_dimensions": [],
            "symmetry": "none",
            "material_thickness": "",
            "tolerances": "",
            "notes": "",
            "raw_text": text[:500]
        }
        
        text_lower = text.lower()
        
        # Extract drawing type
        if "isometric" in text_lower:
            analysis["drawing_type"] = "isometric"
        elif "section" in text_lower:
            analysis["drawing_type"] = "section"
        elif "assembly" in text_lower:
            analysis["drawing_type"] = "assembly"
            analysis["is_assembly"] = True
        
        # Extract dimensions - simple pattern matching
        dimension_patterns = [
            r'(\d+\.?\d*)\s*(mm|cm|m|in|inch|")',  # Values with units
            r'([a-zA-Z\s]+):\s*(\d+\.?\d*)\s*(mm|cm|m|in)',  # Feature: value unit
        ]
        
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 3:  # Feature: value unit
                    feature, value, unit = match
                    clean_feature = feature.strip().replace(' ', '_').lower()
                    analysis["dimensions"][clean_feature] = f"{value} {unit}"
                elif len(match) == 2:  # value unit
                    value, unit = match
                    analysis["dimensions"][f"dim_{len(analysis['dimensions'])}"] = f"{value} {unit}"
        
        # Common feature detection
        features_map = {
            "holes": ["hole", "drill", "bore", "opening"],
            "slots": ["slot", "groove", "channel", "keyway"],
            "fillets": ["fillet", "round", "radius", "curve"],
            "chamfers": ["chamfer", "bevel", "angle"],
            "threads": ["thread", "screw", "bolt"],
            "patterns": ["pattern", "array", "repeat", "circular"],
            "bosses": ["boss", "protrusion", "standoff"],
            "ribs": ["rib", "web", "support"],
            "pockets": ["pocket", "recess", "cavity", "depression"],
        }
        
        for feature_name, keywords in features_map.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis["features"].append(feature_name)
        
        # If no features found, add default
        if not analysis["features"]:
            analysis["features"] = ["extrude"]
        
        # Complexity detection
        if "complex" in text_lower or "complicated" in text_lower:
            analysis["complexity"] = "complex"
        elif "simple" in text_lower or "basic" in text_lower:
            analysis["complexity"] = "simple"
        
        # Assembly detection
        assembly_keywords = ["assembly", "multiple parts", "mating", "fit together", 
                           "component", "sub-assembly", "coupling", "gear"]
        if any(keyword in text_lower for keyword in assembly_keywords):
            analysis["is_assembly"] = True
        
        # Try to extract construction strategy
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if ("start" in line_lower or "begin" in line_lower or 
                "first" in line_lower or "base" in line_lower):
                analysis["construction_strategy"] = line.strip()
                break
        
        if not analysis["construction_strategy"]:
            analysis["construction_strategy"] = "Start with base shape and extrude, then add holes and features"
        
        return analysis
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when everything fails"""
        logger.warning("Using default analysis due to failure")
        return {
            "drawing_type": "orthographic",
            "views": ["front", "top", "side"],
            "dimensions": {
                "length": "100",
                "width": "50", 
                "height": "25",
                "hole_diameter": "10"
            },
            "features": ["extrude", "holes"],
            "complexity": "moderate",
            "is_assembly": False,
            "construction_strategy": "Start with rectangular base from front view, extrude to width, then add holes from top view",
            "critical_dimensions": ["length", "width", "height"],
            "symmetry": "none",
            "material_thickness": "5",
            "tolerances": "±0.5",
            "notes": "Using default analysis - actual drawing may vary",
            "raw_text": ""
        }