import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import threading
import base64
import zipfile
from pathlib import Path
import shutil
import os
import io
import yaml
import time
from datetime import datetime
from dist.src.util import config_util
from dist.src.process import proc

# Session states and global vars

if "score_threshold" not in st.session_state:
    st.session_state.score_threshold = 3.0
if "info_expand" not in st.session_state:
    st.session_state.info_expand = False
if "default_dataset" not in st.session_state:
    st.session_state.default_dataset = ""
if "default_dataset_index" not in st.session_state:
    st.session_state.default_dataset_index = 0
if "passed_lock" not in st.session_state:
    st.session_state.passed_lock = False
if "saliency_extraction_method" not in st.session_state:
    st.session_state.saliency_extraction_method = 0
if "pdsm_method" not in st.session_state:
    st.session_state.pdsm_method = 0

SALIENCY_EXTRACTION_METHODS = ["GradCAM", "Raw", "Flow", "Rollout"]
PDSM_POOLING_METHODS = ["l2_norm", "l1_norm", "max", "sum", "mean", "median"]

INPUTS_DIR = Path("./inputs/")
INPUTS_DIR.mkdir(exist_ok=True)
existing_datasets = [d.name for d in INPUTS_DIR.iterdir() if d.is_dir()]

OUTUTS_DIR = Path("./outputs/")
OUTUTS_DIR.mkdir(exist_ok=True)

# Helper functions

def get_zip_buffer(directory_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for file in Path(directory_path).rglob("*"):
            if file.is_file():
                z.write(file, arcname=file.relative_to(directory_path))
    buf.seek(0)
    return buf


def force_download(zip_buffer, filename):
    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    dl_link = f"""
    <html>
    <body>
        <a id="download_link" href="data:application/zip;base64,{b64}" download="{filename}"></a>
        <script>
            document.getElementById('download_link').click();
        </script>
    </body>
    </html>
    """
    components.html(dl_link, height=0)
    return 0


@st.cache_resource
def get_gpu_lock():
    return threading.Lock()


gpu_lock = get_gpu_lock()


def eval_wrapper(config_file):
    with gpu_lock:    
        st.session_state.passed_lock = True
        rtn_message = proc.run_eval(config_file)
    if rtn_message == 0:
        st.session_state.eval_result = "SUCCESS"
    else:
        st.session_state.error_message = str(rtn_message)
        st.session_state.eval_result = "FAILED"
    st.session_state.eval_done = True


@st.dialog("Processing Data", dismissible=False)
def run_evaluation_dialog(config_file):
    zip_download_name = ""
    st.session_state.eval_done = False
    st.session_state.eval_result = None
    st.session_state.passed_lock = False
    
    thread = threading.Thread(target=eval_wrapper, args=(config_file,))
    add_script_run_ctx(thread, get_script_run_ctx())
    thread.start()

    status_placeholder = st.empty()

    @st.fragment(run_every=2)
    def monitor_progress():
        if st.session_state.eval_done:
            if st.session_state.eval_result == "SUCCESS":
                st.success("Evaluation complete and downloaded started")
                zip_download_name = f"{selected_dataset}_{config_file["datetime"].strftime("%Y%m%d_%H%M")}.zip"

            else:
                st.error(f"Unable to run analysis, refer to generated log file for more information due to error: {st.session_state.error_message}")
                zip_download_name = f"FAILED_{selected_dataset}_{config_file["datetime"].strftime("%Y%m%d_%H%M")}.zip"            

            zip_buffer = get_zip_buffer(os.path.join(f"{OUTUTS_DIR}/{selected_dataset}/", config_file["datetime"].strftime("%Y%m%d_%H%M")))
            force_download(zip_buffer, f"{zip_download_name}")

            if st.session_state.eval_result == "SUCCESS":
                time.sleep(2)
            else:
                time.sleep(8) # give user more time to read error message
            st.rerun() # Closes dialog

        else:
            with status_placeholder.container():
                if st.session_state.passed_lock:
                    with st.spinner('Running evaluation (keep this tab open, it may take a while)...'):
                        time.sleep(2.5) 
                else:
                    with st.spinner('In the queue for resources (keep this tab open, it may take a while)...'):
                        time.sleep(2.5) 

    monitor_progress()


def get_default_dataset_index():
    if st.session_state.default_dataset in existing_datasets:
        st.session_state.default_dataset_index = existing_datasets.index(st.session_state.default_dataset)
    else:
        st.session_state.default_dataset_index = 0


@st.dialog("Uploading Data", dismissible=False)
def run_upload_dialog(uploaded_file):
    with st.spinner('Uploading zip (keep this tab open)...'):
        zip_name = Path(uploaded_file.name).stem
        extract_path = Path(f"{INPUTS_DIR}/{zip_name}")

        # remove data if already exist
        if extract_path.exists() and extract_path.is_dir():
            shutil.rmtree(extract_path)
        os.mkdir(extract_path)

        # extract all from zip file regardless of folder hierarchy
        with zipfile.ZipFile(uploaded_file, 'r') as input_zip:
            for file_info in input_zip.infolist():
                if file_info.filename.lower().endswith('.wav') and not file_info.is_dir():
                    filename_only = os.path.basename(file_info.filename)
                    with open(os.path.join(extract_path, filename_only), "wb") as f:
                        f.write(input_zip.read(file_info.filename))

        # Find how many wav files there are
        wav_files = list(extract_path.rglob("*.wav"))

        if len(wav_files) > 0:
            st.info(f"Uploaded dataset {zip_name}, with {len(wav_files)} wav files")
            st.session_state.default_dataset = zip_name
            time.sleep(2)
        else:
            st.error("Validation failed: No .wav files found in the uploaded zip.")
            if extract_path.exists() and extract_path.is_dir():
                shutil.rmtree(extract_path)
            time.sleep(4)

        st.rerun()  # Programmatically close the dialog and refresh the page


@st.fragment(run_every=2)
def keep_alive():
    pass # keeps websocket open

keep_alive()


# Main page

st.set_page_config(page_title="Synthetic Speech Evaluation Tool")
st.markdown("""
# Synthetic Speech Evaluation Tool
---""")

with st.expander("Information", key=st.session_state.info_expand, expanded=st.session_state.info_expand):
    st.markdown("""
    ### About
    This tool conducts batch analysis of synthetic speech datasets using estimated sound quality scores. After pressing "Run Evaluation", a zip folder containg the results will be downloaded, this contains tab-delimted csv files, system-level plots and individidual audio file plots (for speech samples that fall under the threshold chosen) for further analysis.
       
    ### Usage Notes
    **To use this tool you may do one of the following:**
    - Upload a zip file containing all of your .wav files
    - Select a previously uploaded dataset and download an old analysis (ie: if you lose connectivty to the app) or rerun with a new threshold
            
    ### Threshold Notes
    **When using the threshold, it is worth considering the trade-off between speed and amount of thresholded data:**
    - Scores of each sound quality metric are from 1 (bad) to 5 (good), and the threshold will run analysis on any speech segment that falls below this threshold value for the given sound quality metric.
    - A lower threshold, increases speed and extracts the worst of the dataset (recommended to start).
    - A higher threshold, is slower and more thorough, running analysis on better quality data.
    - If you are unsure, run at threshold of 3 to start. Depending on the dataset if no data is thresholded, consider increasing by 0.5 until a manageable amount of data is thresholded.  

    ### Analysis Reference 
    - "Dimension" refers to the sound quality metric out of "Mean Opinion Score (MOS), Noisiness, Discontinuity, Coloration, and Loudness.
    - Each "segment" of speech refers to a section of an audio file spliced into manageable segments (less than 10 seconds).
    - "Saliency" refers to the important areas of the input spectrogram for the given dimension, as determined by the estimator.
    - "Channel" refers to the wav channel of the input audio file
    """)

uploaded_file = st.file_uploader("Upload new dataset (.zip containing only wav files)", accept_multiple_files=False, type=["zip"])

if st.button("Upload"):

    # check zip file is valid
    if uploaded_file is not None:
        run_upload_dialog(uploaded_file)
    else:
        st.warning("Please upload a file")

if len(existing_datasets) > 0:
    get_default_dataset_index()

    selected_dataset = st.selectbox(
        "Choose a dataset to run analysis on",
        options=existing_datasets, index=st.session_state.default_dataset_index, on_change=get_default_dataset_index
    )

    existing_outputs = None
    selected_output_history = None
    if (selected_dataset != "") and (os.path.exists(os.path.join(OUTUTS_DIR, selected_dataset))):
        existing_outputs = [d.name for d in Path(os.path.join(OUTUTS_DIR, selected_dataset)).iterdir() if d.is_dir()]
        
        with st.expander("History", expanded=False):
            if len(existing_outputs) > 0:
                selected_output_history = st.selectbox(
                    "Previous Evaluations of this Dataset",
                    options=existing_outputs, index=len(existing_outputs)-1, # default to most recent analysis of this dataset if it exists
                    help="Download a previous evaluation of the dataset selected (for instance if you are running with different analysis options or get disconnected during a session)"
                )
                
                if selected_output_history != "":
                    
                    if os.path.exists(os.path.join(f"{OUTUTS_DIR}/{selected_dataset}/config_used.yaml")):
                        with open(os.path.join(f"{OUTUTS_DIR}/{selected_dataset}/config_used.yaml"), 'r') as f:
                            config_temp = yaml.safe_load(f)  
                            if "score_threshold" in config_temp.keys():
                                score_threshold = config_temp["score_threshold"]

                if st.button("Download Previous Evaluation"):    
                    if selected_output_history == "":
                        st.error("Please select a previous dataset evaluation to download")
                    else:
                        try:
                            zip_buffer = get_zip_buffer(os.path.join(f"{OUTUTS_DIR}/{selected_dataset}/", selected_output_history))
                            force_download(zip_buffer, f"{selected_dataset}_{selected_output_history}.zip")
                        except:
                            pass

    score_threshold = st.number_input("Threshold", min_value=1.0, max_value=5.0, step=0.1, value=st.session_state.score_threshold, help="Upper threshold to run analysis on for each sound quality metric.")
    saliency_method = st.selectbox(
        "Saliency Extraction Method",
        options=SALIENCY_EXTRACTION_METHODS, index=st.session_state.saliency_extraction_method,
        help="Method to use for attention saliency extraction from model (GradCAM as default as is perceived to work best for most datasets tested)"
    )
    pdsm_method = st.selectbox(
        "Saliency Extraction Method",
        options=PDSM_POOLING_METHODS, index=st.session_state.pdsm_method,
        help="Method to use in determining the troublesome phonemes in utterance (l2_norm as default due to robustness for many datasets tested)"
    )


    if st.button("Run Evaluation"):
        if selected_dataset == "":
            st.error("Please select a dataset to run analysis on")
        else:
            st.info(f"Running analysis on {selected_dataset} dataset, when complete, your browser will download a zip file containing the results")
            config_data = {
                "dataset_name" : selected_dataset,
                "score_threshold" : score_threshold,
                "datetime" : datetime.now(),
                "pdsm" : {
                    "pool" : pdsm_method
                },
                "saliency" : {
                    "saliency_method" : saliency_method
                }
            }

            # load default config
            config_file = {}
            config_path = "configs/default.yaml"
            with open(config_path, 'r') as f:
                config_file = yaml.safe_load(f)

            config_file = config_util.deep_merge(config_file, config_data)
            run_evaluation_dialog(config_file)

