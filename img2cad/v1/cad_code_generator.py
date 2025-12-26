import re
import textwrap
from typing import Any, Union, Dict

from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

from ..chat_models import MODEL_TYPE, ChatModelParameters
from ..image import ImageData


def _parse_code(input: dict) -> dict:
    match = re.search(r"```(?:python)?\n(.*?)\n```", input["text"], re.DOTALL)
    if match:
        code_output = match.group(1).strip()
        return {"result": code_output}
    else:
        return {"result": None}


_cad_query_examples = [
    (
        "Simple rectangular part with holes - Basic example",
        """import cadquery as cq

# Parameters from drawing
length = 100.0
width = 50.0
thickness = 10.0
hole_diameter = 8.0
hole_spacing = 70.0

# Create base
result = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z")
    .workplane()
    .rarray(hole_spacing, 30, 2, 2)
    .hole(hole_diameter)
    .edges("|Z")
    .fillet(5.0)
)"""
    ),
    (
        "Cylindrical part with complex features",
        """import cadquery as cq
import math

# Parameters
outer_diameter = 60.0
inner_diameter = 40.0
height = 30.0
flange_thickness = 5.0
bolt_hole_diameter = 6.0
bolt_circle_diameter = 50.0

# Create main cylinder
result = (
    cq.Workplane("XY")
    .circle(outer_diameter/2)
    .extrude(height)
    .faces(">Z")
    .workplane()
    .hole(inner_diameter)
    .faces("<Z")
    .workplane()
    .circle(outer_diameter/2 + 10)
    .extrude(flange_thickness)
    .faces("<Z")
    .workplane()
    .circle(bolt_circle_diameter/2)
    .vertices(6)
    .hole(bolt_hole_diameter)
    .edges("|Z")
    .fillet(2.0)
)"""
    ),
    (
        "Oldham coupling - Complex assembly example",
        """import cadquery as cq

# Parameters for Oldham coupling
flange_diameter = 80.0
flange_thickness = 12.0
center_disc_diameter = 50.0
center_disc_thickness = 10.0
tongue_width = 15.0
tongue_height = 8.0
slot_clearance = 0.3
mounting_hole_diameter = 8.0
mounting_hole_count = 6

def create_flange(with_tongue=True, tongue_direction='x'):
    '''Create one flange with optional tongue'''
    # Main flange
    flange = (
        cq.Workplane("XY")
        .circle(flange_diameter/2)
        .extrude(flange_thickness)
    )
    
    # Mounting holes
    flange = (
        flange.faces(">Z")
        .workplane()
        .circle(flange_diameter * 0.4)
        .vertices(mounting_hole_count)
        .hole(mounting_hole_diameter)
    )
    
    # Add tongue if needed
    if with_tongue:
        tongue = (
            cq.Workplane("XY")
            .workplane(offset=flange_thickness)
            .center(0, 0)
            .rect(tongue_width, tongue_width + slot_clearance)
            .extrude(tongue_height)
        )
        
        if tongue_direction == 'y':
            tongue = tongue.rotate((0,0,0), (0,0,1), 90)
        
        flange = flange.union(tongue)
    
    return flange

def create_center_disc():
    '''Create center disc with crossed slots'''
    disc = (
        cq.Workplane("XY")
        .circle(center_disc_diameter/2)
        .extrude(center_disc_thickness)
    )
    
    # X-direction slot
    slot_x = (
        cq.Workplane("XZ")
        .center(0, center_disc_thickness/2)
        .rect(tongue_width + slot_clearance, center_disc_thickness)
        .extrude(center_disc_diameter/2, both=True)
    )
    
    # Y-direction slot
    slot_y = (
        cq.Workplane("YZ")
        .center(0, center_disc_thickness/2)
        .rect(tongue_width + slot_clearance, center_disc_thickness)
        .extrude(center_disc_diameter/2, both=True)
    )
    
    # Cut slots
    disc = disc.cut(slot_x).cut(slot_y)
    
    return disc

# Create all parts
flange1 = create_flange(with_tongue=True, tongue_direction='x')
flange2 = create_flange(with_tongue=True, tongue_direction='y')
center = create_center_disc()

# Position for assembly (optional)
# flange2 = flange2.translate((0, 0, flange_thickness + tongue_height + center_disc_thickness))
# center = center.translate((0, 0, flange_thickness + tongue_height))

result = flange1  # Export main part"""
    ),
    (
        "Multi-view part interpretation",
        """import cadquery as cq

# Dimensions from different views
# Front view: main profile
height = 60.0
width = 40.0
base_thickness = 10.0

# Top view: slot dimensions
slot_length = 30.0
slot_width = 12.0
slot_depth = 5.0

# Side view: hole dimensions
hole_diameter = 8.0
hole_height = 25.0
hole_offset = 15.0

# Create from front view profile
result = (
    cq.Workplane("XZ")  # Front view plane
    .center(0, height/2)
    .rect(width, height)
    .extrude(base_thickness, both=True)  # Extrude in Y direction
)

# Add slot from top view
slot = (
    cq.Workplane("XY")
    .workplane(offset=height/2 - slot_depth)
    .center(0, 0)
    .slot2D(slot_length, slot_width, 0)
    .cutBlind(-slot_depth)
)
result = result.cut(slot)

# Add holes from side view
hole = (
    cq.Workplane("YZ")
    .workplane(offset=width/2 - hole_offset)
    .center(hole_height, 0)
    .circle(hole_diameter/2)
    .cutThruAll()
)
result = result.cut(hole)

# Add fillets
result = result.edges("|Y").fillet(2.0)"""
    ),
    (
        "Gear creation example",
        """import cadquery as cq
import math

# Gear parameters
module = 2.0
num_teeth = 20
pressure_angle = 20.0
thickness = 10.0
bore_diameter = 10.0

def gear_tooth_profile(module, num_teeth, pressure_angle):
    '''Generate gear tooth profile'''
    pitch_diameter = module * num_teeth
    base_diameter = pitch_diameter * math.cos(math.radians(pressure_angle))
    
    # Simplified involute approximation
    points = []
    for i in range(21):
        angle = i * math.pi / (num_teeth * 10)
        r = base_diameter/2 * (1 + angle**2 / 2)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
    
    return points

# Create gear
points = gear_tooth_profile(module, num_teeth, pressure_angle)
result = (
    cq.Workplane("XY")
    .polyline(points)
    .close()
    .revolve(360)
    .faces("<Z")
    .workplane()
    .hole(bore_diameter)
)"""
    ),
]



_design_steps =  """
## UNIVERSAL CAD DESIGN STEPS FOR ANY 2D DRAWING:

### 1. DRAWING ANALYSIS PHASE
   - **Identify Drawing Type**: Determine if it's orthographic, isometric, section, or assembly
   - **Extract All Dimensions**: Get every dimension from the drawing
   - **Understand Views**: Identify front, top, side, isometric views and their relationships
   - **Recognize Features**: Identify holes, slots, fillets, chamfers, threads, patterns
   - **Determine Assembly Relationships**: If multiple parts, understand how they fit

### 2. PARAMETRIC DESIGN SETUP
   - **Define All Parameters**: Create variables for EVERY dimension
   - **Set Up Constants**: Define material thicknesses, clearances, tolerances
   - **Create Functions**: For repetitive features or complex geometry
   - **Establish Workplanes**: Based on drawing views

### 3. BASE FEATURE CONSTRUCTION
   - **Choose Starting Point**: Usually the largest/main feature
   - **Select Workplane**: Based on primary view (usually XY for top view)
   - **Create Base Shape**: Rectangle, circle, polygon, or complex profile
   - **Extrude/Revolve**: Convert 2D profile to 3D

### 4. FEATURE ADDITION ORDER
   - **Primary Features First**: Major cuts, holes, slots
   - **Secondary Features Next**: Smaller details, patterns
   - **Fillets/Chamfers Last**: Always add these after all other features
   - **Symmetry Handling**: Use mirror() for symmetrical parts

### 5. ASSEMBLY HANDLING (if needed)
   - **Create Parts Separately**: Make each component as individual function
   - **Define Mating Features**: Ensure parts fit together properly
   - **Add Clearances**: Include appropriate gaps for movement/fit
   - **Position Components**: Use translate() and rotate() to assemble

### 6. VALIDATION AND REFINEMENT
   - **Check Dimensions**: Verify all dimensions match drawing
   - **Test Clearances**: Ensure moving parts have space
   - **Validate Geometry**: Check for errors or invalid shapes
   - **Optimize Code**: Make it parametric and reusable

### 7. SPECIAL TECHNIQUES FOR COMPLEX PARTS
   - **Multi-View Integration**: Combine information from different views
   - **Parametric Curves**: For gears, cams, complex profiles
   - **Lofting/Sweeping**: For organic or transitioning shapes
   - **Boolean Operations**: Union, cut, intersect for complex forms

### 8. CADQUERY SPECIFIC BEST PRACTICES
   - **Method Chaining Order**: Workplane → Sketch → Operation
   - **Selector Usage**: Use faces(), edges(), vertices() accurately
   - **Workplane Management**: Change workplanes as needed
   - **Error Handling**: Check .isValid() and handle failures
"""


class CadCodeGeneratorChain(SequentialChain):
    model_type: MODEL_TYPE = "gpt"

    def __init__(self, model_type: MODEL_TYPE = "gpt") -> None:
        sample_codes = "\n\n".join(
            [f"{explanation}\n```python\n{code}\n```" for explanation, code in _cad_query_examples]
        )
        gen_cad_code_prompt = (
            "You are a world-class CAD engineer. Create EXACT 3D CadQuery code from ANY 2D CAD drawing.\n\n"
            
            "## DRAWING ANALYSIS SUMMARY:\n"
            "{drawing_analysis}\n\n"
            
            "## DESIGN STEPS TO FOLLOW:\n"
            f"{textwrap.indent(_design_steps, '  ')}\n\n"
            
            "## CRITICAL REQUIREMENTS:\n"
            "1. **USE EXACT DIMENSIONS** from analysis above\n"
            "2. **FOLLOW THE DESIGN STEPS** precisely\n"
            "3. **CREATE PARAMETRIC CODE** with variables for ALL dimensions\n"
            "4. **HANDLE ASSEMBLIES** properly if drawing shows multiple parts\n"
            "5. **INCLUDE CLEARANCES** for moving/fitting parts\n"
            "6. **ADD COMMENTS** explaining each major operation\n"
            "7. **USE BEST PRACTICES** from examples below\n\n"
            
            "## OUTPUT SPECIFICATIONS:\n"
            "* Output file path MUST use: `cq.exporters.export(result, \"${{output_filename}}\")`\n"
            "* Use `${{output_filename}}` template string\n"
            "* Code MUST be inside markdown code block\n"
            "* Include ALL necessary imports\n"
            "* Add proper error handling if needed\n\n"
            
            "## CADQUERY EXAMPLES FOR REFERENCE:\n"
            f"{sample_codes}\n\n"
            
            "## BASED ON THE DRAWING ANALYSIS ABOVE, GENERATE COMPLETE CODE:\n"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": gen_cad_code_prompt},
                {"type": "image_url", "image_url": {"url": "data:image/{image_type};base64,{image_data}"}},
            ]),
        ])
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
            input_variables=["drawing_analysis", "image_type", "image_data"],
            output_variables=["result"],
            verbose=True,
        )
        self.model_type = model_type

    def prep_inputs(self, inputs: Union[dict[str, Any], Any]) -> dict[str, str]:
        """Prepare inputs for the chain"""
        assert isinstance(inputs, dict), "Inputs must be a dictionary"
        
        required_keys = ["drawing_analysis", "image_data"]
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required key: {key}")
        
        image_data = inputs["image_data"]
        if not isinstance(image_data, ImageData):
            raise ValueError("image_data must be ImageData")
        
        # Format drawing analysis
        if isinstance(inputs["drawing_analysis"], dict):
            analysis_text = self._format_analysis(inputs["drawing_analysis"])
        else:
            analysis_text = str(inputs["drawing_analysis"])
        
        # Handle image format for Claude
        if self.model_type == "claude" and image_data.type != "png":
            image_data = image_data.convert("png")
        
        return {
            "drawing_analysis": analysis_text,
            "image_type": image_data.type,
            "image_data": image_data.data
        }
    
    def _format_analysis(self, analysis: Dict) -> str:
        """Format analysis dictionary for prompt"""
        parts = []
        
        parts.append(f"=== DRAWING ANALYSIS ===")
        parts.append(f"Type: {analysis.get('drawing_type', 'Unknown')}")
        parts.append(f"Views: {', '.join(analysis.get('views', []))}")
        parts.append(f"Complexity: {analysis.get('complexity', 'Unknown')}")
        parts.append(f"Is Assembly: {analysis.get('is_assembly', False)}")
        
        if analysis.get('dimensions'):
            parts.append(f"\n=== DIMENSIONS ===")
            for feature, value in analysis['dimensions'].items():
                parts.append(f"- {feature}: {value}")
        
        if analysis.get('features'):
            parts.append(f"\n=== FEATURES ===")
            parts.append(f"{', '.join(analysis['features'])}")
        
        if analysis.get('construction_strategy'):
            parts.append(f"\n=== CONSTRUCTION STRATEGY ===")
            parts.append(analysis['construction_strategy'])
        
        if analysis.get('critical_dimensions'):
            parts.append(f"\n=== CRITICAL DIMENSIONS ===")
            parts.append(f"{', '.join(analysis['critical_dimensions'])}")
        
        if analysis.get('symmetry') and analysis['symmetry'] != 'none':
            parts.append(f"\n=== SYMMETRY ===")
            parts.append(analysis['symmetry'])
        
        if analysis.get('notes'):
            parts.append(f"\n=== NOTES ===")
            parts.append(analysis['notes'])
        
        return "\n".join(parts)
