import tempfile
import os
from string import Template
from typing import Dict, Optional, List
import traceback
from pathlib import Path
import shutil

from loguru import logger

from .agents import execute_python_code
from .chat_models import MODEL_TYPE
from .image import ImageData
from .render import render_and_export_image
from .v1.drawing_analyzer import DrawingAnalyzer
from .v1.cad_code_generator import CadCodeGeneratorChain
from .v1.cad_code_refiner import UniversalCadCodeRefiner
from .v1.assembly_segmentor import segment_assembly_parts


def index_map(index: int) -> str:
    """Convert index to ordinal string"""
    if index == 0:
        return "1st"
    elif index == 1:
        return "2nd"
    elif index == 2:
        return "3rd"
    else:
        return f"{index + 1}th"





class UniversalCADPipeline:
    """Universal pipeline for ANY 2D CAD image"""
    
    def __init__(self, model_type: MODEL_TYPE = "gpt"):
        self.model_type = model_type
        self.analyzer = DrawingAnalyzer(model_type)
        self.generator = CadCodeGeneratorChain(model_type)
        self.refiner = UniversalCadCodeRefiner(model_type)
    
    def process_single_image(self, image_filepath: str, output_filepath: str,
                            max_refinements: int = 3, analysis: Dict = None) -> bool:
        """
        Process a single 2D CAD image and generate 3D STEP file
        """
        logger.info(f"🚀 Processing single part: {image_filepath}")
        
        try:
            # 1. Load image
            image_data = ImageData.load_from_file(image_filepath)
            logger.info(f"📸 Loaded image: {image_filepath} ({image_data.type})")
            
            # 2. Analyze the drawing if not provided
            if analysis is None:
                logger.info("🔍 Analyzing drawing...")
                analysis = self.analyzer.analyze(image_data)
            
            self._log_analysis(analysis)
            
            # 3. Generate initial code
            logger.info("💻 Generating initial CAD code...")
            generated_code = self._generate_initial_code(image_data, analysis, output_filepath)
            if not generated_code:
                logger.error("❌ Failed to generate initial code")
                return False
            
            # 4. Execute and refine
            success = self._execute_and_refine_loop(
                code=generated_code,
                original_image=image_data,
                output_filepath=output_filepath,
                max_refinements=max_refinements,
                analysis=analysis
            )
            
            if success:
                logger.success(f"✅ Successfully created: {output_filepath}")
                if os.path.exists(output_filepath):
                    file_size = os.path.getsize(output_filepath)
                    logger.info(f"📦 File size: {file_size:,} bytes")
            else:
                logger.error(f"❌ Failed to refine: {output_filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"💥 Critical error in single image processing: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def process(self, image_filepath: str, output_filepath: str, 
                max_refinements: int = 3) -> Dict[str, bool]:
        """
        Process ANY 2D CAD image and generate 3D STEP file(s)
        
        Returns: Dictionary mapping part names to success status
        """
        logger.info(f"🚀 Starting universal CAD conversion: {image_filepath}")
        
        try:
            # 1. Load and analyze the main image
            image_data = ImageData.load_from_file(image_filepath)
            logger.info(f"📸 Loaded image: {image_filepath} ({image_data.type})")
            
            # 2. Analyze the drawing
            logger.info("🔍 Analyzing drawing...")
            analysis = self.analyzer.analyze(image_data)
            self._log_analysis(analysis)
            
            results = {}
            
            # 3. Check if assembly
            if analysis.get('drawing_type') == "assembly" or analysis.get('is_assembly'):
                logger.info("🔄 Detected assembly drawing, processing individual parts")
                
                # Define output directories
                # base_dir = Path(image_filepath).parent
                extracted_parts_dir = r"extracted_parts"
                step_files_dir = r"step_files"
                
                # Create directories
                os.makedirs(extracted_parts_dir, exist_ok=True)
                os.makedirs(step_files_dir, exist_ok=True)
                
                # Segment assembly into parts
                part_image_paths = segment_assembly_parts(str(image_filepath), str(extracted_parts_dir))
                logger.info(f"✅ segmented parts saved to: {part_image_paths}")
                
                if not part_image_paths:
                    logger.warning("⚠️ No parts extracted, falling back to single part processing")
                    success = self.process_single_image(image_filepath, step_files_dir, max_refinements, analysis)
                    results["main_part"] = success
                else:
                    logger.info(f"✅ Successfully segmented {len(part_image_paths)} parts")
                    
                    # Process each part
                    for i, part_image_path in enumerate(part_image_paths):
                        part_name = f"part_{i+1}"
                        part_step_path = os.path.join(step_files_dir, f"{part_name}.step")
                        
                        logger.info(f"🔧 Processing {part_name} from: {part_image_path}")
                        
                        # Analyze this specific part
                        part_image_data = ImageData.load_from_file(part_image_path)
                        part_analysis = self.analyzer.analyze(part_image_data)
                        
                        # Process the part
                        success = self.process_single_image(
                            image_filepath=str(part_image_path),
                            output_filepath=str(part_step_path),
                            max_refinements=max_refinements,
                            analysis=part_analysis
                        )
                        
                        results[part_name] = success
                        
                        if success:
                            logger.success(f"✅ Created {part_name} at: {part_step_path}")
                        else:
                            logger.error(f"❌ Failed to create {part_name}")
                         
            else:
                # Single part processing
                logger.info("🔧 Processing as single part")
                success = self.process_single_image(image_filepath, output_filepath, max_refinements, analysis)
                results["main_part"] = success
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Critical error in pipeline: {e}")
            logger.error(traceback.format_exc())
            return {"error": False}
    
    def _log_analysis(self, analysis: Dict):
        """Log analysis results"""
        logger.info(f"📊 Drawing Type: {analysis.get('drawing_type', 'Unknown')}")
        logger.info(f"📊 Complexity: {analysis.get('complexity', 'Unknown')}")
        logger.info(f"📊 Is Assembly: {analysis.get('is_assembly', False)}")
        logger.info(f"📊 Views: {', '.join(analysis.get('views', []))}")
        
        if analysis.get('dimensions'):
            logger.info(f"📊 Dimensions extracted: {len(analysis['dimensions'])}")
        
        if analysis.get('features'):
            logger.info(f"📊 Features: {', '.join(analysis['features'][:5])}")
            if len(analysis['features']) > 5:
                logger.info(f"📊 ... and {len(analysis['features']) - 5} more")
    
    def _generate_initial_code(self, image_data: ImageData, analysis: Dict, 
                              output_filepath: str) -> Optional[str]:
        """Generate initial CAD code"""
        try:
            result = self.generator.invoke({
                "drawing_analysis": analysis,
                "image_data": image_data
            })["result"]
            
            if result:
                code = Template(result).substitute(output_filename=output_filepath)
                logger.debug(f"Generated initial code (first 500 chars):\n{code[:500]}...")
                return code
            else:
                logger.error("No code generated by LLM")
                return None
                
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return None
    
    def _execute_and_refine_loop(self, code: str, original_image: ImageData,
                                output_filepath: str, max_refinements: int,
                                analysis: Dict) -> bool:
        """Execute code and refine through iterations"""
        
        only_execute = (self.model_type == "llama")
        current_code = code
        
        for iteration in range(max_refinements + 1):
            iteration_name = self._iteration_name(iteration)
            logger.info(f"🔄 {iteration_name} iteration")
            
            # Execute current code
            execution_success, execution_output = self._execute_code(
                current_code, only_execute, iteration_name
            )
            
            if execution_success:
                # Check if output file was created
                if os.path.exists(output_filepath):
                    logger.success(f"✓ {iteration_name} execution successful!")
                    
                    # Check if we need more refinement
                    if iteration < max_refinements:
                        needs_refinement = self._check_refinement_needed(
                            output_filepath, original_image, analysis, iteration
                        )
                        
                        if needs_refinement:
                            logger.info("🔄 Model needs refinement, continuing...")
                            refined_code = self._refine_code(
                                current_code, original_image, output_filepath, analysis
                            )
                            
                            if refined_code:
                                current_code = Template(refined_code).substitute(
                                    output_filename=output_filepath
                                )
                                logger.info(f"🔄 Refined code for next iteration")
                                continue
                            else:
                                logger.warning("⚠️ Failed to refine code, keeping current")
                                return True  # Current model is acceptable
                        else:
                            logger.success("✅ Model quality satisfactory!")
                            return True
                    else:
                        logger.info("✅ Max refinements reached")
                        return True
                else:
                    logger.warning(f"⚠️ No output file created in {iteration_name} iteration")
            else:
                logger.warning(f"⚠️ {iteration_name} execution failed")
            
            # If we get here and there are refinements left, try to refine
            if iteration < max_refinements:
                refined_code = self._refine_code(
                    current_code, original_image, output_filepath, analysis
                )
                if refined_code:
                    current_code = Template(refined_code).substitute(
                        output_filename=output_filepath
                    )
                    logger.info(f"🔄 Trying refined code in next iteration")
                else:
                    logger.error("❌ Failed to refine code")
                    return False
            else:
                logger.error("❌ Max refinements reached without success")
                return False
        
        return False
    
    def _execute_code(self, code: str, only_execute: bool, iteration_name: str):
        """Execute Python code and return success status and output"""
        try:
            output = execute_python_code(
                code, 
                model_type=self.model_type, 
                only_execute=only_execute
            )
            
            # Check for common error patterns
            if "error" in output.lower() or "traceback" in output.lower():
                logger.warning(f"⚠️ Errors in {iteration_name} execution:\n{output[:300]}...")
                return False, output
            else:
                logger.debug(f"{iteration_name} execution output:\n{output[:200]}...")
                return True, output
                
        except Exception as e:
            logger.error(f"❌ Exception in {iteration_name} execution: {e}")
            return False, str(e)
    
    def _check_refinement_needed(self, model_path: str, original_image: ImageData,
                                analysis: Dict, iteration: int) -> bool:
        """Determine if model needs more refinement"""
        
        # For first iteration, always refine at least once
        if iteration == 0:
            return True
        
        # Create rendered image for comparison
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            render_path = f.name
        
        try:
            render_and_export_image(model_path, render_path)
            
            # Simple heuristic checks
            needs_refine = False
            
            # Check 1: File size too small for complex drawing
            file_size = os.path.getsize(model_path)
            complexity = analysis.get("complexity", "unknown")
            
            if complexity == "complex" and file_size < 50000:  # 50KB threshold
                logger.info("🔄 Model seems too simple for complex drawing")
                needs_refine = True
            
            # Check 2: Assembly drawing but likely single part
            if analysis.get("is_assembly", False) and file_size < 100000:
                logger.info("🔄 Assembly drawing but small file size")
                needs_refine = True
            
            # Check 3: Many features in analysis but few in model
            feature_count = len(analysis.get("features", []))
            if feature_count > 5 and file_size < 30000:
                logger.info("🔄 Many features expected but small model")
                needs_refine = True
            
            return needs_refine
            
        except Exception as e:
            logger.error(f"❌ Error checking refinement needs: {e}")
            return True  # When in doubt, refine
            
        finally:
            if os.path.exists(render_path):
                os.unlink(render_path)
    
    def _refine_code(self, current_code: str, original_image: ImageData,
                    model_path: str, analysis: Dict) -> Optional[str]:
        """Refine the code based on rendered model"""
        
        # Create rendered image for comparison
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            render_path = f.name
        
        try:
            if os.path.exists(model_path):
                render_and_export_image(model_path, render_path)
                rendered_image = ImageData.load_from_file(render_path)
            else:
                # If no model exists, use original image
                logger.warning("⚠️ No model to render, using original for comparison")
                rendered_image = original_image
            
            # Refine code
            result = self.refiner.invoke({
                "code": current_code,
                "drawing_analysis": analysis,
                "original_input": original_image,
                "rendered_result": rendered_image
            })["result"]
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during refinement: {e}")
            return None
            
        finally:
            if os.path.exists(render_path):
                os.unlink(render_path)
    
    def _iteration_name(self, iteration: int) -> str:
        """Get human-readable iteration name"""
        if iteration == 0:
            return "Initial"
        elif iteration == 1:
            return "1st refinement"
        elif iteration == 2:
            return "2nd refinement"
        elif iteration == 3:
            return "3rd refinement"
        else:
            return f"{iteration}th refinement"


# Main function for backward compatibility
def generate_step_from_2d_cad_image(
    image_filepath: str,
    output_filepath: str,
    num_refinements: int = 3,
    model_type: MODEL_TYPE = "gpt",
):
    """
    Universal function to generate STEP file from ANY 2D CAD image
    
    Args:
        image_filepath (str): Path to ANY 2D CAD image
        output_filepath (str): Path to output STEP file
        num_refinements (int): Number of refinement iterations
        model_type (str): AI model to use
    
    Returns:
        Dictionary with processing results for all parts
    """
    pipeline = UniversalCADPipeline(model_type=model_type)
    results = pipeline.process(
        image_filepath=image_filepath,
        output_filepath=output_filepath,
        max_refinements=num_refinements
    )
    
    # Log summary
    logger.info("📋 Processing Summary:")
    for part_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"  {part_name}: {status}")
    
    return results