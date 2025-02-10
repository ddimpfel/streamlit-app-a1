import zipfile
from PIL import Image
import os
import tempfile
import streamlit as st
    
def get_img_dim(data_dir) -> tuple[int, int]:
    class_path = os.path.join(data_dir, os.listdir(data_dir)[0])
    img_path = os.path.join(data_dir, os.path.join(class_path, os.listdir(class_path)[0]))
    img = Image.open(img_path)
    return img.width, img.height

def extract_data(tmp_dir, dataset_file):
    zip_path = os.path.join(tmp_dir, "dataset.zip")
            
    # Save the uploaded zip file to a temporary directory
    with open(zip_path, "wb") as file:
        file.write(dataset_file.getbuffer())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    
    extracted_dirs = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))]
    if len(extracted_dirs) == 0:
        st.error("No folder found in the zip file.")
        return
    elif len(extracted_dirs) > 1:
        st.error("The zip file should have one master folder. Each class should be represented with folders beneath it.")
        return
    return os.path.join(tmp_dir, extracted_dirs[0])
    

def save_model(model_name, model) -> None:
    """Save a Keras model to a temporary file and provide a download button."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    # Save the model in H5 format
    model.save(tmp_path, save_format='h5')
    with open(tmp_path, "rb") as f:
        model_bytes = f.read()
    os.remove(tmp_path)
    st.download_button(
        label=f"Download {model_name} Model",
        data=model_bytes,
        file_name=f"{model_name}.h5",
        mime="application/octet-stream"
    )
    
def highlight_incorrect(row):
    styles = []
    for col in row.index:
        if col == 'Predicted Label':
            if row['True Label'] != row['Predicted Label']:
                styles.append('color: red; background-color: rgb(245, 245, 245)')
            else:
                styles.append('color: green; background-color: rgb(245, 245, 245)')
        else:
            styles.append('background-color: rgb(245, 245, 245)')
    return styles

def style_df(df):
    return df.style.apply(highlight_incorrect, axis=1)
    
def css():
    st.markdown("""
    <style>    
    .token.token.operator {
        color: white;
    }
	.stTabs [data-baseweb="tab"], .st-emotion-cache-14553y9 {
		height: 50px;
        font-size: 20px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    
    </style>
    """, unsafe_allow_html=True)