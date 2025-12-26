from img2cad import generate_step_from_2d_cad_image
import argparse
    
def main():
    parser = argparse.ArgumentParser(
        description="Universal 2D CAD to 3D Converter - Works with ANY 2D CAD drawing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s simple_part.png                    # Convert simple part
  %(prog)s assembly_drawing.jpg               # Convert assembly
  %(prog)s oldham_coupling.png --refinements 5  # Complex part with more refinements
  %(prog)s multi_view.dwg --model gpt         # Use GPT-4 model
        
Supported:
  - ANY 2D CAD format: PNG, JPG, JPEG, GIF, BMP
  - ANY drawing type: Orthographic, Isometric, Section, Assembly
  - ANY complexity: Simple parts to complex mechanical assemblies
        
Models: gpt (recommended), claude, gemini, llama
        """
    )
    
    parser.add_argument("image_filepath", 
                       help="Path to ANY 2D CAD image")
    
    parser.add_argument("--output", 
                       type=str, 
                       default="output.step",
                       help="Output STEP file path (default: output.step)")
    
    parser.add_argument("--refinements", 
                       type=int, 
                       default=3,
                       help="Refinement iterations (1-10, default: 3)")
    
    parser.add_argument("--model", 
                       type=str, 
                       choices=["gpt", "claude", "gemini", "llama"],
                       default="gpt",
                       help="AI model (default: gpt)")
    
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose debugging output")
    
    args = parser.parse_args()
    generate_step_from_2d_cad_image(args.image_filepath, args.output)


if __name__ == "__main__":
    main()
