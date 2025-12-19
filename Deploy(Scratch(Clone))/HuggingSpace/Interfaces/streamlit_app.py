import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image
import cv2
import tempfile
import time
import tempfile


"""
AERIAL RECONNAISSANCE SYSTEM - COMMAND DASHBOARD v1.0
Copyright (c) 2025 Bayu Ardiyansyah. All Rights Reserved.

Comprehensive tactical dashboard for Military Aircraft Detection.
Features advanced telemetry, model comparison via radar charts, and 
real-time inference capabilities.

Author: Bayu Ardiyansyah (bayuardi30@outlook.com)
"""

st.set_page_config(
    page_title="ARS // COMMAND CENTER",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_tactical_styles():
    """
    Injects CSS for a high-density, professional military interface.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap');

        /* PERBAIKAN DI SINI: Padding Top diperbesar agar tidak ketutupan Header */
        .block-container {
            padding-top: 5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .stApp {
            background-color: #0b0d11; 
            color: #aeb9cc;
            font-family: 'Roboto Mono', monospace;
            font-size: 11px;
        }

        /* Scrollbar Styling (Opsional: Agar scrollbar terlihat lebih tactical) */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0b0d11; 
        }
        ::-webkit-scrollbar-thumb {
            background: #30363d; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #58a6ff; 
        }

        h1 { font-size: 20px !important; color: #e6edf3; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; border-bottom: 1px solid #30363d; padding-bottom: 5px; margin-bottom: 10px; }
        h2 { font-size: 16px !important; color: #58a6ff; letter-spacing: 1px; text-transform: uppercase; margin-top: 10px; margin-bottom: 5px; }
        h3 { font-size: 12px !important; color: #8b949e; text-transform: uppercase; font-weight: 600; margin: 0px; }
        h4 { font-size: 11px !important; color: #d29922; text-transform: uppercase; font-weight: 700; border-left: 3px solid #d29922; padding-left: 8px; margin-top: 10px;}
        p, div, label, span { font-size: 11px !important; }

        section[data-testid="stSidebar"] {
            background-color: #010409;
            border-right: 1px solid #30363d;
        }
        
        div[data-testid="metric-container"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 5px 10px;
            border-left: 3px solid #58a6ff;
            height: 70px;
        }
        label[data-testid="stMetricLabel"] { font-size: 9px !important; color: #8b949e; letter-spacing: 1px; }
        div[data-testid="stMetricValue"] { font-size: 18px !important; color: #e6edf3; }

        div.stDataFrame { border: 1px solid #30363d; }
        
        .stButton > button {
            background-color: #1f6feb;
            color: white;
            border: none;
            border-radius: 0px;
            padding: 8px 16px;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: bold;
            transition: background 0.2s;
        }
        .stButton > button:hover { background-color: #388bfd; }

        .status-box { padding: 10px; border: 1px solid #30363d; margin-bottom: 10px; background: #0d1117; }
        .status-header { color: #58a6ff; font-weight: bold; font-size: 11px; margin-bottom: 5px; }
        
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { height: 30px; padding: 0 10px; font-size: 10px; }

        .footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: #010409; color: #484f58;
            text-align: right; padding: 2px 20px;
            font-size: 9px !important; border-top: 1px solid #30363d; z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)

inject_tactical_styles()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "Reports_csv")
IMAGES_DIR = os.path.join(ROOT_DIR, "Images")
MODELS_DIR = os.path.join(ROOT_DIR, "Models")

def get_clean_model_name(raw_name):
    """Normalize model names to filename safe format."""
    return raw_name.replace(' ', '_').replace('(', '').replace(')', '')

@st.cache_data
def fetch_telemetry_data():
    comp_path = os.path.join(REPORTS_DIR, "Final_Model_Comparison.csv")
    ind_path = os.path.join(REPORTS_DIR, "Industrial_Metrics_Summary.csv")

    df_comp, df_ind = None, None
    confused_data = {}

    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path, sep=",")
        df_comp.columns = df_comp.columns.str.strip()

        if "Model" in df_comp.columns:
            df_comp["Model"] = df_comp["Model"].astype(str).str.strip()
        else:
            df_comp = None  

    if os.path.exists(ind_path):
        df_ind = pd.read_csv(ind_path, sep=",")
        df_ind.columns = df_ind.columns.str.strip()

        if "Model" in df_ind.columns:
            df_ind["Model"] = df_ind["Model"].astype(str).str.strip()

    if df_comp is not None:
        for model_name in df_comp["Model"].unique():
            if "Custom" in model_name:
                pair_file = "Confused_Pairs_Custom_CNN.csv"
            elif "EfficientNet" in model_name:
                pair_file = "Confused_Pairs_EfficientNetB0.csv"
            elif "MobileNet" in model_name:
                pair_file = "Confused_Pairs_MobileNetV2.csv"
            else:
                continue

            pair_path = os.path.join(REPORTS_DIR, pair_file)
            if os.path.exists(pair_path):
                confused_data[model_name] = pd.read_csv(pair_path, sep=",")

    return df_comp, df_ind, confused_data



@st.cache_data
def fetch_detailed_reports(model_name):
    
    base_name = ""
    if "Custom" in model_name:
        base_name = "Custom_CNN_(Base)"
    elif "EfficientNet" in model_name:
        base_name = "EfficientNetB0_(Fine-Tuned)"
    elif "MobileNet" in model_name:
        base_name = "MobileNetV2_Fine-Tuned"

    report_path = os.path.join(REPORTS_DIR, f"{base_name}_report.csv")
    cm_path = os.path.join(IMAGES_DIR, f"{base_name}_confusion_matrix.png")
    hist_path = os.path.join(IMAGES_DIR, f"{base_name}_history.png")

    df_report = pd.read_csv(report_path) if os.path.exists(report_path) else None
    if df_report is not None:
        if 'Unnamed: 0' in df_report.columns:
            df_report.rename(columns={'Unnamed: 0': 'Class'}, inplace=True)
            df_report.set_index('Class', inplace=True)

    return df_report, cm_path, hist_path

@st.cache_data
def get_system_metadata():
    path = os.path.join(REPORTS_DIR, "Custom_CNN_(Base)_report.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        exclude = ['accuracy', 'macro avg', 'weighted avg']
        return [idx for idx in df.index if idx not in exclude]
    return []

@st.cache_resource
def load_neural_core(model_name):
    filename_map = {
        "Custom CNN": "model_custom_cnn.h5",
        "MobileNetV2 (TL)": "model_mobilenetv2_finetuned.h5",
        "EfficientNetB0 (TL)": "model_efficientnet_finetuned.h5"
    }
    filename = filename_map.get(model_name)
    if not filename: return None
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path): return None
    return tf.keras.models.load_model(path)

def plot_radar_comparison(df_comp, df_ind):
    if df_comp is None: return None
    
    df_comp['Model'] = df_comp['Model'].astype(str).str.strip()
    
    if df_ind is not None:
        df_ind['Model'] = df_ind['Model'].astype(str).str.strip()
        df_merged = pd.merge(df_comp, df_ind, on="Model", how="left")
    else:
        df_merged = df_comp.copy()
        
    if 'Inference Time (ms/img)' in df_merged.columns:
        df_merged['Inference Time (ms/img)'].fillna(df_merged['Inference Time (ms/img)'].max(), inplace=True)
    if 'Size (MB)' in df_merged.columns:
        df_merged['Size (MB)'].fillna(df_merged['Size (MB)'].max(), inplace=True)

    df_norm = df_merged.copy()
    categories = ['Accuracy', 'F1-Score (Weighted)', 'Inference Speed', 'Storage Efficiency']
    
    for col in ['Inference Time (ms/img)', 'Size (MB)']:
        if col in df_norm.columns:
            min_v = df_norm[col].min()
            max_v = df_norm[col].max()
            if max_v != min_v and not pd.isna(max_v):
                df_norm[col] = 1 - ((df_norm[col] - min_v) / (max_v - min_v))
            else:
                df_norm[col] = 0.5 
        else:
             df_norm[col] = 0.0 
             
    custom_colors = ['#1f6feb', '#238636', '#d29922', '#a371f7']
    fig = go.Figure()

    for index, row in df_norm.iterrows():
        acc = row['Accuracy'] if pd.notna(row['Accuracy']) else 0
        f1 = row['F1-Score (Weighted)'] if pd.notna(row['F1-Score (Weighted)']) else 0
        speed = row['Inference Time (ms/img)']
        size = row['Size (MB)']

        values = [acc, f1, speed, size]
        values += values[:1] 
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Model'],
            line=dict(color=custom_colors[index % len(custom_colors)]),
            
            hoverlabel=dict(
                bgcolor="#161b22",       
                font_color="#3b7db6",    
                font_family="Roboto Mono",
                bordercolor="#30363d" 
            )
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#30363d'),
            bgcolor='rgba(0,0,0,0)',
            gridshape='linear'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Roboto Mono', color='#c0c0c0', size=9),
        margin=dict(l=40, r=40, t=40, b=40), 
        height=250, 
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.3)
    )
    return fig

def render_sidebar_controls():
    with st.sidebar:
        st.markdown("### SYSTEM CONTROLS")
        app_mode = st.radio("SELECT MODULE", ["Analytics Dashboard", "Live Inference"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### DATA TELEMETRY")
        df_comp, _, _ = fetch_telemetry_data()
        
        if df_comp is not None:
            st.success(f"STATUS: ONLINE ({len(df_comp)} MODELS LOADED)")
            st.code("\n".join(df_comp['Model'].tolist()), language="text")
        else:
            st.error("DATA SOURCE: OFFLINE")
            
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 10px; color: #8b949e;">
            OPERATOR: BAYU ARDIYANSYAH<br>
            ID: BAYUARDI30<br>
            SECURE CONNECTION
            </div>
            """, 
            unsafe_allow_html=True
        )
        return app_mode

def render_analytics(df_comp, df_ind, confused_data):
    st.title("MISSION ANALYTICS // MODEL PERFORMANCE")
    
    if df_comp is None or df_ind is None:
        st.warning("AWAITING DATA INGESTION... PLEASE RUN TRAINING PIPELINE.")
        return

    best_model_idx = df_comp['F1-Score (Weighted)'].idxmax()
    best_model = df_comp.loc[best_model_idx]
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("TOP ARCHITECTURE", best_model['Model'])
    kpi2.metric("PEAK ACCURACY", f"{best_model['Accuracy']:.2%}")
    kpi3.metric("F1-SCORE", f"{best_model['F1-Score (Weighted)']:.4f}")
    kpi4.metric("EVALUATED UNITS", len(df_comp))
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown("### MULTI-DIMENSIONAL COMPARISON")
        st.caption("Comparing All Architectures on Accuracy, Speed, and Size.")
        fig_radar = plot_radar_comparison(df_comp, df_ind)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c_right:
        st.markdown("### ACCURACY & F1-SCORE BENCHMARK")
        fig_bar = px.bar(
            df_comp, 
            x='Model', 
            y=['Accuracy', 'F1-Score (Weighted)'],
            barmode='group',
            color_discrete_sequence=['#238636', '#1f6feb'],
            height=280
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Roboto Mono', color='#c0c0c0', size=9),
            margin=dict(l=20, r=20, t=5, b=10),
            legend=dict(orientation="h", y=1.3)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### COMPUTATIONAL COST MATRIX")
    fig_scatter = px.scatter(
        df_ind,
        x='Inference Time (ms/img)',
        y='Size (MB)',
        color='Model',
        size_max=15,
        symbol='Model',
        text='Model',
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=300
    )
    fig_scatter.update_traces(marker=dict(size=12, line=dict(width=1, color='white')), textposition='top center')
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(14, 17, 23, 0.5)',
        font=dict(family='Roboto Mono', color='#c0c0c0', size=9),
        xaxis_title="LATENCY (MS) - LOWER IS BETTER",
        yaxis_title="SIZE (MB) - LOWER IS BETTER",
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.header("DEEP DIVE // MODEL FORENSICS")
    
    available_models = df_comp['Model'].unique()
    
    c_sel, c_blank = st.columns([1, 2])
    with c_sel:
        selected_drilldown = st.selectbox("SELECT TARGET ARCHITECTURE", available_models)
    
    df_report, cm_path, hist_path = fetch_detailed_reports(selected_drilldown)
    
    tab1, tab2, tab3 = st.tabs(["CLASSIFICATION REPORT", "TRAINING DYNAMICS", "CONFUSION ANALYSIS"])
    
    with tab1:
        st.markdown(f"#### PERFORMANCE METRICS: {selected_drilldown}")
        if df_report is not None:
            st.dataframe(
                df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                use_container_width=True,
                height=250
            )
        else:
            st.warning(f"REPORT FILE NOT FOUND: {selected_drilldown}")

    with tab2:
        st.markdown(f"#### LOSS & ACCURACY CURVES: {selected_drilldown}")
        if os.path.exists(hist_path):
            st.image(Image.open(hist_path), use_container_width=True, caption="Training & Validation History")
        else:
            st.warning("GRAPHIC ASSET NOT FOUND.")
            
    with tab3:
        c_cm, c_pairs = st.columns([1.5, 1])
        
        with c_cm:
            st.markdown("#### CONFUSION MATRIX HEATMAP")
            if os.path.exists(cm_path):
                st.image(Image.open(cm_path), use_container_width=True, caption=f"Confusion Matrix: {selected_drilldown}")
            else:
                st.warning("CONFUSION MATRIX IMAGE NOT FOUND.")
        
        with c_pairs:
            st.markdown("#### TOP CONFUSION PAIRS")
            if selected_drilldown in confused_data:
                df_pairs = confused_data[selected_drilldown]
                st.dataframe(
                    df_pairs, 
                    column_config={
                        "Count": st.column_config.ProgressColumn(
                            "Misclassifications",
                            format="%d",
                            min_value=0,
                            max_value=int(df_pairs['Count'].max()) if not df_pairs.empty else 1,
                        )
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )
            else:
                st.info("NO CONFUSION PAIR DATA AVAILABLE FOR THIS MODEL.")


def render_inference_engine():
    st.title("LIVE RECONNAISSANCE // INFERENCE ENGINE")
    
    col_input, col_output = st.columns([1, 2], gap="medium")
    
    with col_input:
        st.markdown("### CONFIGURATION")
        model_options = ["Custom CNN", "MobileNetV2 (TL)", "EfficientNetB0 (TL)"]
        
        df_comp, _, _ = fetch_telemetry_data()
        default_idx = 2 
        if df_comp is not None:
            best = df_comp.loc[df_comp['F1-Score (Weighted)'].idxmax()]['Model']
            if best in model_options:
                default_idx = model_options.index(best)
                
        selected_model = st.selectbox("ACTIVE ARCHITECTURE", model_options, index=default_idx)
        
        # UPDATE: Support Video & Image
        uploaded_file = st.file_uploader("UPLOAD SOURCE (IMG/VIDEO)", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
        
        st.markdown("### SYSTEM LOG")
        if uploaded_file:
            st.text(f"FILE: {uploaded_file.name}")
            st.text(f"TYPE: {uploaded_file.type}")
        else:
            st.text("STATUS: IDLE")

    with col_output:
        st.markdown("### TACTICAL FEED")
        
        if uploaded_file:
            # Load Model & Labels
            class_labels = get_system_metadata()
            model = load_neural_core(selected_model)
            
            if not class_labels or not model:
                st.error("SYSTEM ERROR: Model/Metadata Missing.")
                return

            # --- LOGIKA UNTUK GAMBAR (IMAGE) ---
            if uploaded_file.type.startswith('image'):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Streamlit butuh RGB
                
                # 1. Preprocess
                img_resized = cv2.resize(frame, (224, 224))
                img_batch = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)
                
                # 2. Predict
                preds = model.predict(img_batch)
                score = np.max(preds)
                label = class_labels[np.argmax(preds)]
                
                # 3. Draw Tactical Box (HUD Style)
                # Karena ini Classification Model (bukan Object Detection), 
                # kita buat 'Target Lock' di tengah frame.
                h, w, _ = frame.shape
                color = (35, 134, 54) if score > 0.75 else (210, 153, 34) # Green or Yellow
                
                # Draw Box
                cv2.rectangle(frame, (50, 50), (w-50, h-50), color, 2) # Frame luar
                cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), color, 1) # Crosshair H
                cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), color, 1) # Crosshair V
                
                # Draw Label Background
                text = f"{label}: {score:.2%}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (50, 50 - 30), (50 + text_w, 50), color, -1)
                cv2.putText(frame, text, (50, 50 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                st.image(frame, use_container_width=True, caption="TARGET IDENTIFIED")
                
                # Tampilkan Grafik Probabilitas di bawah gambar
                st.markdown("#### SIGNAL ANALYSIS")
                top_5_idx = preds[0].argsort()[-5:][::-1]
                probs_df = pd.DataFrame({
                    'Class': [class_labels[i] for i in top_5_idx],
                    'Probability': preds[0][top_5_idx]
                })
                fig_prob = px.bar(
                    probs_df, y='Class', x='Probability', orientation='h',
                    text_auto='.1%',
                    color='Probability', color_continuous_scale=['#0d1117', '#1f6feb']
                )
                fig_prob.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Roboto Mono', color='#c0c0c0', size=10),
                    yaxis=dict(autorange="reversed"), margin=dict(t=0, b=0, l=0, r=0),
                    height=200, showlegend=False
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            # --- LOGIKA UNTUK VIDEO ---
            elif uploaded_file.type.startswith('video'):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
                st.video(video_path)
            
                status = st.empty()
                metrics = st.columns(2)
            
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                sample_every = int(fps * 2) 
                frame_id = 0
            
                last_label = "UNKNOWN"
                last_score = 0.0
            
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
            
                    if frame_id % sample_every == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(frame_rgb, (224, 224))
                        img_batch = np.expand_dims(
                            img_resized.astype("float32") / 255.0, axis=0
                        )
            
                        preds = model.predict(img_batch, verbose=0)
                        last_score = float(np.max(preds))
                        last_label = class_labels[int(np.argmax(preds))]
            
                        status.markdown(
                            f"**TARGET UPDATE:** `{last_label}` | `{last_score:.2%}`"
                        )
            
                        metrics[0].metric("TARGET", last_label)
                        metrics[1].metric("CONFIDENCE", f"{last_score:.2%}")
            
                    frame_id += 1
                    time.sleep(0.01)
            
                cap.release()


        else:
            st.info("AWAITING DATALINK STREAM...")

def main():
    mode = render_sidebar_controls()
    df_comp, df_ind, confused_data = fetch_telemetry_data()
    
    if mode == "Analytics Dashboard":
        render_analytics(df_comp, df_ind, confused_data)
    elif mode == "Live Inference":
        render_inference_engine()

    st.markdown("""
        <div class="footer">
            SYSTEM: v1.0 // AUTHOR: BAYU ARDIYANSYAH (bayuardi30@outlook.com) // CLASSIFIED
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()