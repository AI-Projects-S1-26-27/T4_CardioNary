"""CardioNary: Multimodal Clinical Screening Dashboard (Streamlit Edge Frontend).

Integrates phonocardiogram heart sound auscultation, chest radiography (X-ray),
and clinical patient metadata through strict core.contracts interfaces.
Strict adherence to the 'NO RED' clinical safety rule.
Bilingual support: English and Hindi (हिंदी).

Run offline:
    streamlit run app/app.py
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import sys
from typing import Optional

# Ensure repository root is on sys.path so core, cardio, and lung can be imported
# when run via `streamlit run app/app.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Core contracts and pipeline calls
from core.contracts import (
    CLASSES,
    FusedOutput,
    ModelOutput,
    PatientMetadata,
    SYMPTOMS,
    VisitDecision,
)
from cardio.predict import analyse_heart
from lung.predict import analyse_xray
TEST_DATA_DIR = REPO_ROOT / "Test_Data"

# ---------------------------------------------------------------------------
# Bilingual Dictionary (English & Hindi)
# ---------------------------------------------------------------------------
I18N = {
    "en": {
        "title": "CardioNary: Multimodal Clinical Screening Station",
        "subtitle": "Offline Edge Triage Assistant • Version 2.0 (Heart & Lung Branches)",
        "sidebar_header": "1. Patient Intake & Vitals",
        "patient_id": "Blinded Patient ID",
        "age": "Age (Years)",
        "sex": "Biological Sex",
        "sex_female": "Female",
        "sex_male": "Male",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "pregnant": "Pregnancy Status",
        "symptoms": "Reported Symptoms",
        "symp_breathless": "Breathlessness",
        "symp_chest_pain": "Chest Pain",
        "symp_cough": "Cough",
        "symp_fever": "Fever",
        "ops_header": "Operational Settings",
        "cost_ratio": "Clinical Cost Ratio (C_miss / C_refer)",
        "cost_ratio_help": "Higher ratio increases sensitivity to minimise missed clinical abnormalities.",
        "threshold_label": "Effective Referral Threshold (τ)",
        "sample_header": "Quick Test Samples",
        "load_heart_sample": "Select Sample Heart Sound",
        "load_lung_sample": "Select Sample Chest X-Ray",
        "tab_heart": "Heart Sound Auscultation",
        "tab_lung": "Chest Radiography (X-Ray)",
        "heart_upload_label": "Upload Digital Stethoscope Recording (.wav)",
        "heart_player_label": "Auscultation Playback",
        "lung_upload_label": "Upload Chest Radiograph (.png, .jpg, .jpeg)",
        "run_btn": "Run Comprehensive Screening Analysis",
        "no_data_warn": "Please provide at least one diagnostic test (Heart Sound or Chest X-Ray) to evaluate.",
        "analyzing": "Running inference through calibrated late-fusion pipeline...",
        "findings_header": "Diagnostic Screening Findings",
        "decision_refer": "Secondary Referral Recommended",
        "decision_refer_desc": "The combined abnormality probability of {p_any:.1%} exceeds the operational threshold of {tau:.1%} (Cost Ratio {cost_ratio:.1f}:1). Advised action: Refer patient for secondary evaluation at specialist clinic.",
        "decision_routine": "Routine Finding / Low Risk",
        "decision_routine_desc": "The combined abnormality probability of {p_any:.1%} is within normal parameters below threshold ({tau:.1%}). Advised action: Continue standard community health follow-up.",
        "heart_eval_header": "Heart Auscultation Evaluation",
        "lung_eval_header": "Chest Radiography Evaluation",
        "calibrated_prob": "Calibrated Abnormality Probability",
        "top_categories": "Condition Probabilities",
        "biomarkers_header": "Acoustic Biomarkers",
        "heart_rate": "Estimated Heart Rate",
        "murmur_timing": "Murmur Phase Attribution",
        "systole": "Systolic Murmur Detected",
        "diastole": "Diastolic Murmur Detected",
        "neither": "No Pathological Murmur Timing",
        "unknown_phase": "Heart Cycles Indeterminate",
        "segmentation_status": "Cycle Segmentation",
        "segmentation_ok": "Satisfactory (S1/S2 distinct)",
        "segmentation_fail": "Low confidence / noisy cycles",
        "energy_check": "High-Frequency Noise Check",
        "withheld_warn": "Heart result withheld: {reason}",
        "reason_too_short": "Audio clip is too short (minimum 1.5 s required)",
        "reason_unreadable": "Audio file format cannot be decoded",
        "reason_quality_check": "Excess high-frequency acoustic noise detected (>1 kHz threshold)",
        "no_usable_test": "No visit decision: no test gave a usable result. See the reason below, then record or capture again.",
        "fusion_breakdown": "Late Fusion Vote Composition",
        "vote_cnn": "CNN Deep Learning Vote",
        "vote_features": "Secondary Features Vote",
        "vote_metadata": "Patient Demographics Vote",
        "weight_label": "weight",
        "texture_biomarkers": "Radiological Texture Biomarkers",
        "dense_area": "Dense Parenchymal Area",
        "asymmetry": "Hemithorax Asymmetry Index",
        "heterogeneity": "Texture Heterogeneity",
        "clahe_label": "Contrast-Enhanced (CLAHE) View",
        "original_img_label": "Original Film",
        "export_btn": "Export Clinical Triage Summary (JSON)",
        "reset_btn": "Start Next Patient (Clear Form)",
    },
    "hi": {
        "title": "कार्डियोनेरी: बहुविध नैदानिक स्क्रीनिंग स्टेशन",
        "subtitle": "ऑफ़लाइन एज ट्राइएज सहायक • संस्करण 2.0 (हृदय एवं फेफड़े प्रभाग)",
        "sidebar_header": "1. रोगी विवरण एवं महत्वपूर्ण लक्षण",
        "patient_id": "गोपनीय रोगी पहचान (ID)",
        "age": "उम्र (वर्ष)",
        "sex": "लिंग",
        "sex_female": "महिला",
        "sex_male": "पुरुष",
        "height": "ऊंचाई (सेमी)",
        "weight": "वजन (किग्रा)",
        "pregnant": "गर्भावस्था स्थिति",
        "symptoms": "दर्ज लक्षण",
        "symp_breathless": "सांस फूलना (Breathlessness)",
        "symp_chest_pain": "छाती में दर्द (Chest Pain)",
        "symp_cough": "खांसी (Cough)",
        "symp_fever": "बुखार (Fever)",
        "ops_header": "संचालन सेटिंग्स (सीमा स्तर)",
        "cost_ratio": "नैदानिक लागत अनुपात (C_miss / C_refer)",
        "cost_ratio_help": "अनुपात बढ़ाने से संवेदनशीलता बढ़ती है ताकि संभावित विकृति न छूटे।",
        "threshold_label": "प्रभावी रेफरल सीमा (τ)",
        "sample_header": "त्वरित परीक्षण नमूने",
        "load_heart_sample": "नमूना हृदय ध्वनि चुनें",
        "load_lung_sample": "नमूना छाती एक्स-रे चुनें",
        "tab_heart": "हृदय ध्वनि ऑस्कल्टेशन",
        "tab_lung": "छाती का एक्स-रे (रेडियोग्राफी)",
        "heart_upload_label": "डिजिटल स्टेथोस्कोप रिकॉर्डिंग (.wav) अपलोड करें",
        "heart_player_label": "ऑडियो प्लेबैक",
        "lung_upload_label": "छाती का एक्स-रे (.png, .jpg) अपलोड करें",
        "run_btn": "विस्तृत स्क्रीनिंग विश्लेषण चलाएं",
        "no_data_warn": "कृपया मूल्यांकन के लिए कम से कम एक नैदानिक परीक्षण (हृदय ध्वनि या एक्स-रे) प्रदान करें।",
        "analyzing": "कैलिब्रेटेड लेट-फ्यूजन पाइपलाइन के माध्यम से विश्लेषण जारी है...",
        "findings_header": "नैदानिक स्क्रीनिंग परिणाम",
        "decision_refer": "द्वितीयक विशेषज्ञ परामर्श की अनुशंसा (रेफरल)",
        "decision_refer_desc": "कुल असामान्य संभावना {p_any:.1%} निर्धारित सीमा {tau:.1%} से अधिक है (लागत अनुपात {cost_ratio:.1f}:1)। सलाह: रोगी को विशेषज्ञ अस्पताल में द्वितीयक जांच के लिए रेफर करें।",
        "decision_routine": "सामान्य स्थिति / कम जोखिम",
        "decision_routine_desc": "कुल असामान्य संभावना {p_any:.1%} सामान्य सीमा ({tau:.1%}) के अंतर्गत है। सलाह: नियमित सामुदायिक स्वास्थ्य निगरानी जारी रखें।",
        "heart_eval_header": "हृदय ध्वनि मूल्यांकन",
        "lung_eval_header": "छाती एक्स-रे मूल्यांकन",
        "calibrated_prob": "कैलिब्रेटेड असामान्यता संभावना",
        "top_categories": "रोग श्रेणीवार संभावनाएं",
        "biomarkers_header": "ध्वनि बायोमार्कर्स (लक्षण)",
        "heart_rate": "अनुमानित हृदय गति (HR)",
        "murmur_timing": "मर्मर चरण (समय)",
        "systole": "सिस्टोलिक मर्मर पाया गया",
        "diastole": "डायस्टोलिक मर्मर पाया गया",
        "neither": "कोई असामान्य मर्मर नहीं",
        "unknown_phase": "हृदय चक्र अस्पष्ट",
        "segmentation_status": "चक्र विभाजन (S1/S2)",
        "segmentation_ok": "संतोषजनक (स्पष्ट चक्र)",
        "segmentation_fail": "शोरगुल / अस्पष्ट चक्र",
        "energy_check": "उच्च आवृत्ति शोर परीक्षण",
        "withheld_warn": "हृदय परिणाम रोका गया: {reason}",
        "reason_too_short": "ऑडियो क्लिप बहुत छोटी है (न्यूनतम 1.5 सेकंड आवश्यक)",
        "reason_unreadable": "ऑडियो फ़ाइल डिकोड नहीं की जा सकी",
        "reason_quality_check": "उच्च आवृत्ति का अत्यधिक शोर पाया गया (>1 kHz)",
        "no_usable_test": "कोई निर्णय नहीं: किसी भी जांच से उपयोगी परिणाम नहीं मिला। नीचे कारण देखें, फिर दोबारा रिकॉर्ड करें।",
        "fusion_breakdown": "लेट फ्यूजन वोट वितरण",
        "vote_cnn": "CNN डीप लर्निंग वोट",
        "vote_features": "ध्वनिक लक्षण (Features) वोट",
        "vote_metadata": "रोगी जनसांख्यिकी वोट",
        "weight_label": "भार (Weight)",
        "texture_biomarkers": "एक्स-रे बनावट बायोमार्कर्स",
        "dense_area": "घना पैरेन्काइमल क्षेत्र",
        "asymmetry": "हेमिथोरेक्स विषमता सूचकांक",
        "heterogeneity": "बनावट विषमता (Heterogeneity)",
        "clahe_label": "कंट्रास्ट-संवर्धित (CLAHE) दृश्य",
        "original_img_label": "मूल एक्स-रे छवि",
        "export_btn": "नैदानिक ट्राइएज सारांश डाउनलोड करें (JSON)",
        "reset_btn": "नया रोगी प्रारंभ करें (फ़ॉर्म साफ़ करें)",
    },
}

# ---------------------------------------------------------------------------
# UI Theme & Styling (Strictly NO RED)
# ---------------------------------------------------------------------------
def apply_custom_styles():
    st.markdown("""
    <style>
        /* Modern clinical styling */
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1a365d;
            margin-bottom: 4px;
        }
        .main-sub {
            font-size: 1.05rem;
            color: #4a5568;
            margin-bottom: 20px;
        }
        /* Metric card */
        .clinical-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #2b5c8f;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 14px;
        }
        /* Referral Box: Warm Amber (NO RED) */
        .refer-box-caution {
            background-color: #fff8e1;
            border: 2px solid #ffd54f;
            border-radius: 10px;
            padding: 20px 24px;
            color: #8c5b00;
            margin-bottom: 20px;
        }
        /* Routine Box: Soft Green */
        .refer-box-normal {
            background-color: #e8f5e9;
            border: 2px solid #81c784;
            border-radius: 10px;
            padding: 20px 24px;
            color: #1b5e20;
            margin-bottom: 20px;
        }
        /* Biomarker badge */
        .bio-badge {
            background-color: #e2e8f0;
            color: #1e293b;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.88rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 6px;
        }
    </style>
    """, unsafe_allow_html=True)


def plot_waveform(signal: np.ndarray, fs: int = 8000) -> plt.Figure:
    """Waveform envelope plot in clinical slate blue."""
    fig, ax = plt.subplots(figsize=(7.5, 2.0), dpi=100)
    t = np.linspace(0, len(signal) / fs, len(signal))
    ax.plot(t, signal, color="#2b5c8f", lw=0.7)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title("Phonocardiogram (PCG) Envelope", fontsize=9, fontweight="bold", color="#1e293b")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    return fig


def load_test_manifest() -> dict:
    """Reads Test_Data/index.json if available to populate test samples."""
    index_file = TEST_DATA_DIR / "index.json"
    if index_file.exists():
        try:
            return json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cardio": {"items": []}, "pulmo": {"items": []}}


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CardioNary | Multimodal Screening",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    # Session State initialisation
    if "sample_audio_path" not in st.session_state:
        st.session_state.sample_audio_path = None
    if "sample_image_path" not in st.session_state:
        st.session_state.sample_image_path = None

    # Language toggle in sidebar
    with st.sidebar:
        st.markdown("### 🌐 Language / भाषा")
        lang_choice = st.radio("Select Language / भाषा चुनें", ["English", "हिंदी"], horizontal=True, label_visibility="collapsed")
        lang = "en" if lang_choice == "English" else "hi"
        t = I18N[lang]

    # App Title & Subtitle
    st.markdown(f'<div class="main-header">{t["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-sub">{t["subtitle"]}</div>', unsafe_allow_html=True)

    manifest = load_test_manifest()

    # -----------------------------------------------------------------------
    # Sidebar: Patient Metadata & Operations
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(f"### {t['sidebar_header']}")
        anon_id = st.text_input(t["patient_id"], value="PT-2026-0816")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(t["age"], min_value=0.0, max_value=120.0, value=12.0, step=1.0)
            height = st.number_input(t["height"], min_value=20.0, max_value=250.0, value=135.0, step=1.0)
        with c2:
            sex_choice = st.selectbox(t["sex"], options=["female", "male"], format_func=lambda s: t["sex_female"] if s == "female" else t["sex_male"])
            weight = st.number_input(t["weight"], min_value=1.0, max_value=300.0, value=32.0, step=0.5)

        pregnant = st.checkbox(t["pregnant"], value=False)

        st.markdown(f"**{t['symptoms']}**")
        symp_breathless = st.checkbox(t["symp_breathless"])
        symp_chest_pain = st.checkbox(t["symp_chest_pain"])
        symp_cough = st.checkbox(t["symp_cough"])
        symp_fever = st.checkbox(t["symp_fever"])

        symptoms = set()
        if symp_breathless: symptoms.add("breathlessness")
        if symp_chest_pain: symptoms.add("chest_pain")
        if symp_cough: symptoms.add("cough")
        if symp_fever: symptoms.add("fever")

        patient_meta = PatientMetadata(
            age_years=float(age),
            sex=sex_choice,
            height_cm=float(height),
            weight_kg=float(weight),
            pregnant=bool(pregnant),
            symptoms=frozenset(symptoms),
        )

        st.divider()
        st.markdown(f"### {t['ops_header']}")
        cost_ratio = st.slider(
            t["cost_ratio"],
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help=t["cost_ratio_help"],
        )
        threshold = 1.0 / (1.0 + cost_ratio)
        st.markdown(f"{t['threshold_label']}: **{threshold:.3f}**")

        st.divider()
        st.markdown(f"### {t['sample_header']}")

        # Sample Heart Audio selector
        cardio_items = manifest.get("cardio", {}).get("items", [])
        if cardio_items:
            sample_options_heart = ["None / Custom Upload"] + [
                f"{item.get('vector_id', i)}: {item.get('ground_truth', 'sample')} ({item.get('file', '')})"
                for i, item in enumerate(cardio_items[:15])
            ]
            chosen_heart = st.selectbox(t["load_heart_sample"], sample_options_heart)
            if chosen_heart != "None / Custom Upload":
                idx = int(chosen_heart.split(":")[0])
                st.session_state.sample_audio_path = str(TEST_DATA_DIR / cardio_items[idx]["file"])
            else:
                st.session_state.sample_audio_path = None

        # Sample Pulmo Image selector
        pulmo_items = manifest.get("pulmo", {}).get("items", [])
        if pulmo_items:
            sample_options_pulmo = ["None / Custom Upload"] + [
                f"{item.get('vector_id', i)}: {item.get('ground_truth', 'sample')} ({item.get('file', '')})"
                for i, item in enumerate(pulmo_items[:15])
            ]
            chosen_pulmo = st.selectbox(t["load_lung_sample"], sample_options_pulmo)
            if chosen_pulmo != "None / Custom Upload":
                idx = int(chosen_pulmo.split(":")[0])
                st.session_state.sample_image_path = str(TEST_DATA_DIR / pulmo_items[idx]["file"])
            else:
                st.session_state.sample_image_path = None

        if st.button(t["reset_btn"], use_container_width=True):
            st.session_state.sample_audio_path = None
            st.session_state.sample_image_path = None
            st.rerun()

    # -----------------------------------------------------------------------
    # Main Tabs: Diagnostics Input
    # -----------------------------------------------------------------------
    tab_heart, tab_lung = st.tabs([f"🫀 {t['tab_heart']}", f"🫁 {t['tab_lung']}"])

    audio_source = None
    with tab_heart:
        st.markdown(f"#### {t['tab_heart']}")
        c_up, c_aud = st.columns([1.5, 1])

        with c_up:
            up_wav = st.file_uploader(t["heart_upload_label"], type=["wav"])
            if up_wav is not None:
                audio_source = up_wav
            elif st.session_state.sample_audio_path:
                st.info(f"Loaded Sample: `{Path(st.session_state.sample_audio_path).name}`")
                audio_source = st.session_state.sample_audio_path

        with c_aud:
            if audio_source is not None:
                st.markdown(f"**{t['heart_player_label']}**")
                if isinstance(audio_source, str):
                    st.audio(audio_source, format="audio/wav")
                else:
                    audio_source.seek(0)
                    st.audio(audio_source, format="audio/wav")
                    audio_source.seek(0)

    image_source = None
    with tab_lung:
        st.markdown(f"#### {t['tab_lung']}")
        c_img_up, c_prev = st.columns([1.5, 1])

        with c_img_up:
            up_img = st.file_uploader(t["lung_upload_label"], type=["png", "jpg", "jpeg", "dcm"])
            if up_img is not None:
                image_source = up_img
            elif st.session_state.sample_image_path:
                st.info(f"Loaded Sample: `{Path(st.session_state.sample_image_path).name}`")
                image_source = st.session_state.sample_image_path

        with c_prev:
            if image_source is not None:
                st.image(image_source, width=220, caption=t["original_img_label"])

    # -----------------------------------------------------------------------
    # Action Button: Run Analysis
    # -----------------------------------------------------------------------
    st.write("")
    analyze_btn = st.button(f"🔍 {t['run_btn']}", type="primary", use_container_width=True)

    if analyze_btn:
        if audio_source is None and image_source is None:
            st.warning(t["no_data_warn"])
            return

        with st.spinner(t["analyzing"]):
            heart_res = None
            lung_res = None
            p_heart = None
            p_xray = None

            # 1. Process Heart Sound
            if audio_source is not None:
                try:
                    heart_res = analyse_heart(audio_source, patient_meta)
                    if heart_res.fused is not None:
                        p_heart = heart_res.fused.p_abnormal
                except Exception as exc:
                    st.warning(f"Heart analysis note: {exc}")

            # 2. Process Chest X-Ray
            clahe_preview = None
            if image_source is not None:
                try:
                    lung_model_out, lung_fused_out, clahe_preview = analyse_xray(image_source, patient_meta)
                    lung_res = (lung_model_out, lung_fused_out)
                    p_xray = lung_fused_out.p_abnormal
                except Exception as exc:
                    st.warning(f"X-ray analysis note: {exc}")

            # 3. Compute Joint Visit Decision. A VisitDecision needs at least one
            # usable test; a withheld heart result with no X-ray gives none.
            decision = None
            if p_heart is not None or p_xray is not None:
                if p_heart is not None and p_xray is not None:
                    p_any = float(1.0 - (1.0 - p_heart) * (1.0 - p_xray))
                elif p_heart is not None:
                    p_any = float(p_heart)
                else:
                    p_any = float(p_xray)
                decision = VisitDecision(
                    refer=bool(p_any >= threshold),
                    p_any=p_any,
                    cost_ratio=float(cost_ratio),
                    p_heart=p_heart,
                    p_xray=p_xray,
                )

        # -------------------------------------------------------------------
        # Presentation of Results: Unified Decision Card
        # -------------------------------------------------------------------
        st.markdown(f"### {t['findings_header']}")

        if decision is None:
            st.info(t["no_usable_test"])
        elif decision.refer:
            msg = t["decision_refer_desc"].format(p_any=decision.p_any, tau=threshold, cost_ratio=cost_ratio)
            st.markdown(f"""
            <div class="refer-box-caution">
                <h3 style="margin-top:0;">⚠️ {t['decision_refer']}</h3>
                <p style="font-size:1.05rem; margin-bottom:0;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            msg = t["decision_routine_desc"].format(p_any=decision.p_any, tau=threshold)
            st.markdown(f"""
            <div class="refer-box-normal">
                <h3 style="margin-top:0;">✅ {t['decision_routine']}</h3>
                <p style="font-size:1.05rem; margin-bottom:0;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

        col_h, col_l = st.columns([1, 1], gap="medium")

        # -------------------------------------------------------------------
        # Heart Branch Results Panel
        # -------------------------------------------------------------------
        with col_h:
            st.markdown(f"#### {t['heart_eval_header']}")
            if heart_res is not None:
                if heart_res.withheld:
                    reason_text = t.get(f"reason_{heart_res.withheld}", heart_res.withheld)
                    st.warning(t["withheld_warn"].format(reason=reason_text))
                elif heart_res.fused is not None and heart_res.model_output is not None:
                    st.metric(
                        t["calibrated_prob"],
                        f"{heart_res.fused.p_abnormal:.1%}",
                        help=f"Temperature calibrated (T = {heart_res.temperature:.3f})" if heart_res.temperature else "Uncalibrated",
                    )

                    st.markdown(f"**{t['top_categories']}**")
                    probs = heart_res.model_output.probabilities
                    df_h = pd.DataFrame({
                        "Condition": list(probs.keys()),
                        "Probability": [f"{v:.1%}" for v in probs.values()],
                    })
                    st.dataframe(df_h, use_container_width=True, hide_index=True)

                    st.markdown(f"**{t['biomarkers_header']}**")
                    ch = heart_res.characteristics
                    hr = ch.get("heart_rate_bpm")
                    seg_ok = ch.get("segmentation_ok")
                    phase = heart_res.systole_or_diastole

                    st.write(f"- {t['heart_rate']}: **{hr:.0f} bpm**" if hr else f"- {t['heart_rate']}: N/A")
                    st.write(f"- {t['segmentation_status']}: **{t['segmentation_ok'] if seg_ok else t['segmentation_fail']}**")
                    
                    if phase == "systole":
                        st.write(f"- {t['murmur_timing']}: **{t['systole']}**")
                    elif phase == "diastole":
                        st.write(f"- {t['murmur_timing']}: **{t['diastole']}**")
                    elif phase == "neither":
                        st.write(f"- {t['murmur_timing']}: **{t['neither']}**")
                    else:
                        st.write(f"- {t['murmur_timing']}: **{t['unknown_phase']}**")

                    st.markdown(f"**{t['fusion_breakdown']}**")
                    fused = heart_res.fused
                    for vote_name, vote_val in fused.votes.items():
                        v_label = t.get(f"vote_{vote_name}", vote_name)
                        w_val = fused.weights.get(vote_name, 0.0)
                        st.write(f"- {v_label}: `{vote_val:.1%}` ({t['weight_label']}: {w_val:.2f})")
            else:
                st.info("Heart sound analysis not performed.")

        # -------------------------------------------------------------------
        # Lung Branch Results Panel
        # -------------------------------------------------------------------
        with col_l:
            st.markdown(f"#### {t['lung_eval_header']}")
            if lung_res is not None:
                l_model_out, l_fused_out = lung_res
                st.metric(
                    t["calibrated_prob"],
                    f"{l_fused_out.p_abnormal:.1%}",
                    help="Kermany calibrated probability",
                )

                st.markdown(f"**{t['top_categories']}**")
                probs_l = l_model_out.probabilities
                df_l = pd.DataFrame({
                    "Condition": list(probs_l.keys()),
                    "Probability": [f"{v:.1%}" for v in probs_l.values()],
                })
                st.dataframe(df_l, use_container_width=True, hide_index=True)

                st.markdown(f"**{t['texture_biomarkers']}**")
                sec = l_model_out.secondary
                dense = sec.get("dense_area_pct")
                asym = sec.get("asymmetry_index")
                het = sec.get("texture_heterogeneity")

                if dense is not None: st.write(f"- {t['dense_area']}: **{dense:.1f}%**")
                if asym is not None: st.write(f"- {t['asymmetry']}: **{asym:.3f}**")
                if het is not None: st.write(f"- {t['heterogeneity']}: **{het:.2f}**")

                if clahe_preview is not None:
                    st.image(clahe_preview, width=220, caption=t["clahe_label"])

                st.markdown(f"**{t['fusion_breakdown']}**")
                for vote_name, vote_val in l_fused_out.votes.items():
                    v_label = t.get(f"vote_{vote_name}", vote_name)
                    w_val = l_fused_out.weights.get(vote_name, 0.0)
                    st.write(f"- {v_label}: `{vote_val:.1%}` ({t['weight_label']}: {w_val:.2f})")
            else:
                st.info("Chest X-ray analysis not performed.")

        # -------------------------------------------------------------------
        # Export Clinical Triage Summary (JSON)
        # -------------------------------------------------------------------
        st.divider()
        summary_payload = {
            "patient_id": anon_id,
            "refer": decision.refer if decision else None,
            "p_any": decision.p_any if decision else None,
            "threshold": threshold,
            "cost_ratio": float(cost_ratio),
            "p_heart": p_heart,
            "p_xray": p_xray,
            "demographics": {
                "age_years": patient_meta.age_years,
                "sex": patient_meta.sex,
                "height_cm": patient_meta.height_cm,
                "weight_kg": patient_meta.weight_kg,
                "pregnant": patient_meta.pregnant,
                "symptoms": list(patient_meta.symptoms),
            },
        }

        st.download_button(
            label=f"📄 {t['export_btn']}",
            data=json.dumps(summary_payload, indent=2),
            file_name=f"{anon_id}_triage_summary.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
