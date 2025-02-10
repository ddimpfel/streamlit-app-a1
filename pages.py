import streamlit as st
import keras
import tempfile
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models import build_model
from utils import save_model, extract_data, get_img_dim, style_df

def home() -> None:
    st.title("Image Classification Model Trainer")
    st.write(
        """
        Test different parameters on basic image classification models to see what could work best.  
        Upload your dataset, choose your parameters, and even download the final models!
        """
    )
    st.write("### Previously Trained Models")
    if "models" in st.session_state and st.session_state.models:
        for model_name, info in st.session_state.models.items():
            if info['test_accuracy'] > 0.7:
                st.write(f"{model_name} – Accuracy: :green[{info['test_accuracy']:.2f}]")
            else:
                st.write(f"{model_name} – Accuracy: :red[{info['test_accuracy']:.2f}]")
    else:
        st.info("No models have been trained yet.")
    if st.button("Start Training New Model", use_container_width=True):
        st.session_state.page = "Train"
        st.rerun()

def train_model() -> None:
    st.title("Train Model")
    
    # User input
    dataset_file = st.file_uploader("Upload Dataset (ZIP file)", type=["zip"])
    col1, col2, col3 = st.columns(3)
    
    with col1:
        val_split = st.number_input("% to use in Validation", value=0.2, step=0.05)
    with col2:
        batch_size = st.number_input("Batch Size", value=16, step=1)
    with col3:
        epochs = st.number_input("Total Training Epochs", value=3, step=1)
        
    if val_split < 0 or val_split > 1:
        st.error("Validation split must be in the range [0, 1].")
        return
    
    if st.button("Start Training"):
        if dataset_file is None:
            st.error("Please upload a dataset file.")
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = extract_data(tmp_dir, dataset_file)
            
            img_width, img_height = get_img_dim(data_dir)
            
            try:
                train_data = keras.preprocessing.image_dataset_from_directory(
                    data_dir,
                    validation_split=val_split,
                    subset="training",
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size
                )
                val_data = keras.preprocessing.image_dataset_from_directory(
                    data_dir,
                    validation_split=val_split,
                    subset="validation",
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size
                )
            except Exception as e:
                st.error("Error creating datasets: " + str(e))
                return

            num_classes = len(train_data.class_names)
            st.write(f"Found {len(train_data.class_names)} classes:", train_data.class_names)

            prog = 0
            progress_bar = st.progress(prog)
            status_text = st.empty()
            
            model_types = ["Basic CNN Model", "Block CNN Model", "Batch CNN Model"]
            results = {}
            models_dict = {}
            
            # Train and evaluate each model
            for model_name in model_types:
                status_text.text(f"Training {model_name}...")
                model = build_model(model_name, (img_height, img_width, 3), num_classes)
                prog+=5
                progress_bar.progress(prog)
                
                start_time = time.time()
                model.fit(train_data, epochs=epochs, validation_data=val_data, verbose=0)
                train_time = time.time() - start_time
                prog+=13
                progress_bar.progress(prog)
                
                loss, accuracy = model.evaluate(val_data, verbose=0)
                
                # Get classification results on validation data
                all_preds = []
                all_true = []
                confidences = []
                for images, labels in val_data:
                    preds = model.predict(images, verbose=0)
                    batch_preds = np.argmax(preds, axis=1)
                    all_preds.extend(batch_preds)
                    all_true.extend(labels.numpy())
                    confidences.extend(np.max(preds, axis=1))
                prog+=5
                progress_bar.progress(prog)
                
                df_results = pd.DataFrame({
                    "Image Index": range(len(all_true)),
                    "True Label": all_true,
                    "Predicted Label": all_preds,
                    "Confidence": confidences
                })
                results_df_styled = style_df(df_results)
                
                results[model_name] = {
                    "train_time": train_time,
                    "test_accuracy": accuracy,
                    "test_loss": loss,
                    "param_count": model.count_params(),
                    "results_df": df_results,
                    "results_df_styled": results_df_styled
                }
                models_dict[model_name] = model
                
                prog+=10
                progress_bar.progress(prog)
            
            progress_bar.progress(prog+1)
            
            # Save the model info in session state
            st.session_state.models = {}
            for model_name in model_types:
                st.session_state.models[model_name] = {
                    "model": models_dict[model_name],
                    "train_time": results[model_name]["train_time"],
                    "test_accuracy": results[model_name]["test_accuracy"],
                    "test_loss": results[model_name]["test_loss"],
                    "param_count": results[model_name]["param_count"],
                    "results_df": results[model_name]["results_df"],
                    "results_df_styled": results[model_name]["results_df_styled"],
                    "class_names": train_data.class_names
                }
            st.session_state.page = "Training Results"
            st.rerun()
    
    st.subheader("NOTE:")
    st.write("Your files directory structure should be as follows")
    st.code("""
                file.zip  
                    - top_directory  
                        - class_1_dir  
                            - images  
                        - class_2_dir  
                            - images  
                        - ...  
                """)
    
def results() -> None:
    st.title("Training Results")

    if "models" not in st.session_state or not st.session_state.models:
        st.error("No trained models found.")
        return

    # Create a tab for each model.
    model_names = list(st.session_state.models.keys())
    tabs = st.tabs(model_names)

    # Display each models training statistics per tab
    for i, mtype in enumerate(model_names):
        with tabs[i]:
            info = st.session_state.models[mtype]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Training Time: {info['train_time']:.2f} seconds")
                st.write(f"Test Loss: {info['test_loss']:.2f}")
            with col2:
                if info['test_accuracy'] > 0.7:
                    st.write(f"Test Accuracy: :green[{info['test_accuracy']:.2f}]")
                else:
                    st.write(f"Test Accuracy: :red[{info['test_accuracy']:.2f}]")
                st.write(f"Total Parameters: {info.get('param_count', 'N/A')}")
            
            if st.button(f"Save {mtype}", key=f"save_{mtype}", type="primary"):
                save_model(mtype, info["model"])
            
            st.write("### Individual Classifications Table")
            styled_df = info["results_df_styled"]
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            df_results = info["results_df"]
            cm = confusion_matrix(df_results["True Label"], df_results["Predicted Label"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
    with col4:
        if st.button("Back to Training", use_container_width=True):
            st.session_state.page = "Train"
            st.rerun()
