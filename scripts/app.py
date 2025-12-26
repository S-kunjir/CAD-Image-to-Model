import argparse
import os
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
from img2cad import generate_step_from_2d_cad_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt", 
                       choices=["gpt", "claude", "gemini", "llama"])
    parser.add_argument("--refinements", type=int, default=3)
    return parser.parse_args()


args = parse_args()

st.set_page_config(
    page_title="Universal 2D CAD to 3D Converter", 
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .info-box {
        background-color: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .assembly-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .part-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin-bottom: 0.5rem;
    }
    .download-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
            
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔄 Universal 2D CAD to 3D Converter</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Convert <b>ANY</b> 2D CAD drawing to 3D STEP file</p>', unsafe_allow_html=True)
st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload 2D CAD Drawing")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Upload ANY 2D CAD drawing in common image formats"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded CAD Drawing", use_column_width=True)
        
        # Image info
        with st.expander("📊 Image Details", expanded=False):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Format", uploaded_file.type.split('/')[-1].upper())
                st.metric("Width", f"{image.size[0]} px")
            with col_info2:
                st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
                st.metric("Height", f"{image.size[1]} px")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.markdown('<div class="info-box">✅ Image uploaded successfully. Ready for conversion.</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📥 Generated 3D Model")
    
    if uploaded_file is not None:
        os.makedirs("input_folder",exist_ok=True)
        input_image_path = os.path.join("input_folder",uploaded_file.name)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        convert_button = st.button(
            "🚀 Convert to 3D", 
            type="primary", 
            use_container_width=True,
            help="Click to start the conversion process"
        )
        
        if convert_button:
            # Create output path
            output_path = "step_files"
            
            # Progress container
            progress_container = st.container()
            status_container = st.container()
            result_container = st.container()
            
            with progress_container:
                st.markdown("### 📈 Conversion Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Update progress callback
            def update_progress(step, total_steps=7):
                progress = (step / total_steps)
                progress_bar.progress(progress)
            
            try:
                # Step 1: Upload
                status_text.text("📤 Step 1/7: Processing upload...")
                update_progress(1)
                
                # Step 2: Analysis
                status_text.text("🔍 Step 2/7: Analyzing drawing...")
                update_progress(2)
                
                # Run conversion
                with st.spinner("Processing... This may take a few minutes for complex drawings."):
                    results = generate_step_from_2d_cad_image(
                        image_filepath=input_image_path,
                        output_filepath=output_path,
                        num_refinements=3,
                        model_type='gpt'
                    )
                
                # Update progress
                update_progress(7)
                status_text.text("✅ Conversion complete!")
                
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
                # Show results based on type
                with result_container:
                    # Check if it's an assembly (multiple parts)
                    is_assembly = len(results) > 1 and any(k.startswith('part_') for k in results.keys())
                    
                    if is_assembly:
                        # ASSEMBLY UI
                        st.markdown("### 🔧 Assembly Detected!")
                        
                        with st.expander("📊 Assembly Summary", expanded=True):
                            st.markdown(f"**Detected {len([k for k in results.keys() if k.startswith('part_')])} individual parts**")
                            
                            # Create results table
                            part_results = []
                            for key, success in results.items():
                                if key.startswith('part_'):
                                    part_num = key.replace('part_', '')
                                    part_results.append({
                                        'Part': f"Part {part_num}",
                                        'Status': '✅ Success' if success else '❌ Failed',
                                        'File': f"part_{part_num}.step"
                                    })
                            
                            if part_results:
                                df = pd.DataFrame(part_results)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Assembly info box
                        st.markdown('<div class="assembly-box">🔧 <b>Assembly Drawing Detected</b><br>Individual STEP files have been generated for each part. You can download them individually below.</div>', unsafe_allow_html=True)
                        
                        # Check where STEP files are saved
                        step_files_dir = Path(r"step_files")
                        
                        if step_files_dir.exists():
                            # Get all STEP files
                            step_files = list(step_files_dir.glob("*.step"))
                            
                            if step_files:
                                st.markdown("### 📥 Download Individual Parts")
                                
                                # Create download grid
                                cols = st.columns(3)
                                for idx, step_file in enumerate(step_files):
                                    with cols[idx % 3]:
                                        with open(step_file, "rb") as f:
                                            file_size = os.path.getsize(step_file)
                                            st.download_button(
                                                label=f"Part {idx+1} ({file_size:,} bytes)",
                                                data=f,
                                                file_name=step_file.name,
                                                mime="application/step",
                                                use_container_width=True
                                            )
                            else:
                                st.warning("No STEP files found in the step_files directory.")
                        
                        
                        
                    else:
                        # SINGLE PART UI
                        st.markdown("### ✅ Single Part Conversion")
                        
                        with st.expander("📊 Conversion Results", expanded=True):
                            col_result1, col_result2 = st.columns(2)
                            with col_result1:
                                st.metric("Status", "Success" if results.get('main_part') else "Failed")
                                st.metric("Type", "Single Part")
                            with col_result2:
                                if os.path.exists(output_path):
                                    file_size = os.path.getsize(output_path)
                                    st.metric("File Size", f"{file_size:,} bytes")
                                st.metric("Refinements", num_refinements)
                        
                        # Single part download
                        if os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            
                            st.markdown('<div class="success-box">🎉 3D model successfully generated! You can now open it in any CAD software (Fusion 360, SolidWorks, FreeCAD, etc.)</div>', unsafe_allow_html=True)
                            
                            # Download button
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label=f"📥 Download 3D Model ({file_size:,} bytes)",
                                    data=file,
                                    file_name="output.step",
                                    mime="application/step",
                                    use_container_width=True
                                )
                            
                            # Preview info
                            with st.expander("🔍 How to Use", expanded=False):
                                st.markdown("""
                                1. **Open in CAD software**: Import the STEP file into Fusion 360, SolidWorks, FreeCAD, etc.
                                2. **Modify if needed**: The model is fully parametric
                                3. **Export to other formats**: Convert to STL for 3D printing
                                4. **Perform analysis**: Use for FEA, simulations, etc.
                                """)
                        else:
                            st.error("❌ Failed to generate 3D model. Please try with more refinements or a different model.")
                        
            except Exception as e:
                st.error(f"❌ Error during conversion: {str(e)}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    else:
        st.markdown("""
        <div class="info-box">
        <h4>👈 Upload a CAD drawing to begin</h4>
        <p>This system automatically detects:</p>
        <ul>
        <li><b>Single parts</b> → One STEP file</li>
        <li><b>Assemblies</b> → Multiple STEP files (one per part)</li>
        </ul>
        <p>Supported drawings:</p>
        <ul>
        <li>2D CAD drawings (single parts or assemblies)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        

st.markdown("---")
