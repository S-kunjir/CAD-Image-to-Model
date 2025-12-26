from typing import Any, Union, Dict
import re

from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

from .cad_code_generator import _parse_code
from ..chat_models import MODEL_TYPE, ChatModelParameters
from ..image import ImageData


class UniversalCadCodeRefiner(SequentialChain):
    """Universal CAD code refiner for any 2D CAD drawing"""
    
    model_type: MODEL_TYPE = "gpt"

    def __init__(self, model_type: MODEL_TYPE = "gpt") -> None:
        # REFINEMENT DESIGN STEPS
        refinement_steps = """
        ## REFINEMENT PROCESS:
        1. **Compare 3D render with original 2D drawing**
        2. **Identify discrepancies** in geometry, dimensions, features
        3. **Check assembly relationships** if multiple parts
        4. **Verify clearances and fits**
        5. **Ensure parametric variables** are used correctly
        6. **Improve code structure and comments**
        
        ## COMMON ISSUES TO FIX:
        - Missing features from drawing
        - Incorrect dimensions
        - Wrong workplane orientation
        - Missing fillets/chamfers
        - Assembly misalignment
        - Inadequate clearances
        - Non-parametric code
        
        ## REFINEMENT PRIORITY:
        1. Fix critical dimensional errors
        2. Add missing major features
        3. Correct assembly relationships
        4. Add fillets/chamfers
        5. Improve parametric design
        6. Enhance code readability
        """
        
        refine_cad_code_prompt = (
            "You are a CAD quality engineer refining 3D code based on comparison with original 2D drawing.\n\n"
            
            "## ORIGINAL DRAWING vs 3D RENDER COMPARISON:\n"
            "Compare these images side by side to identify issues.\n\n"
            
            "## DRAWING ANALYSIS CONTEXT:\n"
            "{drawing_analysis}\n\n"
            
            "## CURRENT CODE (NEEDS REFINEMENT):\n"
            "```python\n"
            "{code}\n"
            "```\n\n"
            
            "## REFINEMENT STEPS TO FOLLOW:\n"
            f"{refinement_steps}\n\n"
            
            "## REFINEMENT INSTRUCTIONS:\n"
            "1. **Compare geometries** - What's different?\n"
            "2. **Check dimensions** - Do they match the drawing?\n"
            "3. **Verify features** - Are all holes, slots, fillets present?\n"
            "4. **Assembly check** - Do parts fit properly?\n"
            "5. **Clearance validation** - Are moving parts correctly spaced?\n"
            "6. **Code improvement** - Make it more parametric/readable\n\n"
            
            "## OUTPUT REQUIREMENTS:\n"
            "* Provide COMPLETE refined code\n"
            "* Keep the `${{output_filename}}` template\n"
            "* Code MUST be in markdown code block\n"
            "* Explain major changes in comments\n"
            "* Maintain backward compatibility\n\n"
            
            "## REFINED CODE:\n"
        )
        
        if model_type in ["gpt", "claude", "gemini"]:
            prompt = ChatPromptTemplate(
                input_variables=[
                    "code",
                    "drawing_analysis",
                    "original_image_type",
                    "original_image_data",
                    "rendered_image_type",
                    "rendered_image_data",
                ],
                messages=[
                    HumanMessagePromptTemplate(
                        prompt=[
                            PromptTemplate(
                                input_variables=["code", "drawing_analysis"],
                                template=refine_cad_code_prompt
                            ),
                            ImagePromptTemplate(
                                input_variables=["original_image_type", "original_image_data"],
                                template={
                                    "url": "data:image/{original_image_type};base64,{original_image_data}",
                                },
                            ),
                            ImagePromptTemplate(
                                input_variables=["rendered_image_type", "rendered_image_data"],
                                template={
                                    "url": "data:image/{rendered_image_type};base64,{rendered_image_data}",
                                },
                            ),
                        ]
                    )
                ],
            )
            input_vars = [
                "code", "drawing_analysis", "original_image_type", 
                "original_image_data", "rendered_image_type", "rendered_image_data"
            ]
        elif model_type == "llama":
            prompt = ChatPromptTemplate(
                input_variables=[
                    "code",
                    "drawing_analysis",
                    "combined_image_type",
                    "combined_image_data",
                ],
                messages=[
                    HumanMessagePromptTemplate(
                        prompt=[
                            PromptTemplate(
                                input_variables=["code", "drawing_analysis"],
                                template=refine_cad_code_prompt
                            ),
                            ImagePromptTemplate(
                                input_variables=["combined_image_type", "combined_image_data"],
                                template={
                                    "url": "data:image/{combined_image_type};base64,{combined_image_data}",
                                },
                            ),
                        ]
                    )
                ],
            )
            input_vars = ["code", "drawing_analysis", "combined_image_type", "combined_image_data"]
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        llm = ChatModelParameters.from_model_name(model_type).create_chat_model()

        super().__init__(
            chains=[
                LLMChain(prompt=prompt, llm=llm),  # type: ignore
                TransformChain(
                    input_variables=["text"],
                    output_variables=["result"],
                    transform=_parse_code,
                    atransform=None,
                ),
            ],
            input_variables=input_vars,
            output_variables=["result"],
            verbose=True,
        )
        self.model_type = model_type

    def prep_inputs(self, inputs: Union[dict[str, Any], Any]) -> dict[str, str]:
        """Prepare inputs for refinement"""
        required_keys = ["code", "drawing_analysis", "original_input", "rendered_result"]
        
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required key: {key}")
        
        original_image = inputs["original_input"]
        rendered_image = inputs["rendered_result"]
        
        if not isinstance(original_image, ImageData):
            raise ValueError("original_input must be ImageData")
        if not isinstance(rendered_image, ImageData):
            raise ValueError("rendered_result must be ImageData")
        
        # Format drawing analysis
        if isinstance(inputs["drawing_analysis"], dict):
            from .cad_code_generator import UniversalCadCodeGenerator
            generator = UniversalCadCodeGenerator(self.model_type)
            analysis_text = generator._format_analysis(inputs["drawing_analysis"])
        else:
            analysis_text = str(inputs["drawing_analysis"])
        
        # Prepare inputs based on model type
        prepared_inputs = {
            "code": str(inputs["code"]),
            "drawing_analysis": analysis_text,
        }
        
        if self.model_type in ["gpt", "claude", "gemini"]:
            # Convert images for Claude if needed
            if self.model_type == "claude":
                if original_image.type != "png":
                    original_image = original_image.convert("png")
                if rendered_image.type != "png":
                    rendered_image = rendered_image.convert("png")
            
            prepared_inputs.update({
                "original_image_type": original_image.type,
                "original_image_data": original_image.data,
                "rendered_image_type": rendered_image.type,
                "rendered_image_data": rendered_image.data,
            })
        
        elif self.model_type == "llama":
            # Combine images for Llama
            combined_image = original_image.merge(rendered_image)
            prepared_inputs.update({
                "combined_image_type": combined_image.type,
                "combined_image_data": combined_image.data,
            })
        
        return prepared_inputs


# Keep original class for backward compatibility
CadCodeRefinerChain = UniversalCadCodeRefiner