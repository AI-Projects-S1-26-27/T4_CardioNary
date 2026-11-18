"""Dashboard text in English and Hindi.

Two sources, looked up in this order:

  APP     CardioNary's own strings: patient details, late fusion, the visit
          decision and the architecture panel.
  CHIRON  the Chiron screening station's interface and clinical reference text,
          app/chiron_i18n.json, exported unchanged from gateway/static/i18n.js.

The Hindi has not been reviewed by a native-speaking clinician. Class names and
clinical guidance come from CHIRON, keyed by the class order in core.contracts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from core.contracts import CLASSES

LANGS = {"en": "English", "hi": "हिंदी"}
CHIRON_FILE = Path(__file__).resolve().with_name("chiron_i18n.json")
CHIRON_NODE = {"heart": "1", "xray": "2"}      # Chiron's node types: 1 cardiac, 2 pulmonary

APP = {
    "en": {
        # header and notices
        "demo_label": "PUBLIC DEMO",
        "public_demo_note": "Use the built-in reference cases or other public data only. Never upload a real "
                            "patient's recording or X-ray. Results are a demonstration, not a diagnosis.",
        "model_ready": "{name} model ready",
        "model_missing": "{name} model missing",
        "dataset_heart": "Yaseen 2018 heart sounds, 5 classes; reference cases from held-out fold 0",
        "dataset_xray": "Kermany 2018 paediatric chest X-rays; reference cases from held-out fold 0",
        "footer_repo": "Source: github.com/AI-Projects-S1-26-27/T4_CardioNary",
        "retention": "Records live only in this browser session and are cleared when it ends. "
                     "Nothing is stored on the server.",

        # architecture panel
        "arch_title": "System architecture",
        "arch_hint": "The highlighted blocks follow what you are doing",
        "cap_start": "Start here: pick a reference case or upload a file in each card below. "
                     "Patient details are optional.",
        "cap_heart": "Heart sound selected. Stepping to another case analyses it.",
        "cap_xray": "Chest X-ray selected. Stepping to another case analyses it.",
        "cap_wait_upload": "Waiting for an upload in the {branch} card.",
        "cap_meta": "Patient details feed the metadata vote of both branches. "
                    "Press Analyse visit to use them.",
        "cap_decision": "The tests included and the cost ratio set the visit decision.",
        "cap_running": "Running: {stage}",
        "cap_done": "Analysis complete. The visit decision and the screening report are below.",
        "cap_branch_done": "{branch} result ready. Press Analyse visit for the combined decision.",
        "cap_batch": "Screening batch: {branch} case {i} of {n}",
        "cap_batch_done": "Screening batch complete: see the batch results and the case log below.",
        "lg_waiting": "Waiting for input",
        "lg_ready": "Ready",
        "lg_active": "Running now",
        "lg_done": "Done",
        "lg_withheld": "Withheld",
        "lg_unused": "Not used",
        "lg_focus": "Where you are",
        "lg_cnn": "Compact CNN (8-bit; adapts at each new clinic)",
        "lg_classic": "Classical processing, metadata, late fusion and decision",

        # stage names shown while running
        "stg_heart_load_and_resample": "reading the heart sound, resampling to 8 kHz",
        "stg_heart_quality_check": "quality check (length, noise above 1 kHz)",
        "stg_heart_front_end": "spectrogram (40 × 192 log-mel)",
        "stg_heart_cnn_8bit": "8-bit heart CNN",
        "stg_heart_secondary_characteristics": "secondary characteristics (heart rate, rhythm, sound timing)",
        "stg_heart_systole_or_diastole": "systole or diastole timing",
        "stg_heart_votes_and_fusion": "votes and late fusion (heart)",
        "stg_xray_load_image": "reading the X-ray as a 96 × 96 grayscale image",
        "stg_xray_cnn_8bit": "8-bit X-ray CNN",
        "stg_xray_secondary_characteristics": "secondary characteristics (contrast, texture)",
        "stg_xray_occlusion_map": "focus region (3 × 3 occlusion map)",
        "stg_xray_votes_and_fusion": "votes and late fusion (X-ray)",
        "stg_decision": "one decision per visit",
        "stg_report": "screening report",

        # architecture blocks: (bold lines, normal lines)
        "blk_heart_audio": (["Heart audio"], ["3 seconds"]),
        "blk_spectrogram": (["Spectrogram"], ["40 × 192"]),
        "blk_heart_cnn": (["Heart CNN"], ["compact, 8-bit", "adapts per clinic"]),
        "blk_heart_secondary": (["Secondary"], ["characteristics", "heart rate, rhythm,", "sound timing"]),
        "blk_heart_features": (["Feature classifier"], ["logistic regression"]),
        "blk_heart_fusion": (["Late fusion"], ["weighted votes"]),
        "blk_metadata": (["Patient metadata and symptoms"], ["age, sex, body size, symptoms"]),
        "blk_meta_classifier": (["Metadata classifier"], ["logistic regression"]),
        "blk_xray": (["Chest X-ray"], ["any size"]),
        "blk_grayscale": (["Grayscale image"], ["96 × 96"]),
        "blk_xray_cnn": (["X-ray CNN"], ["MobileNetV2, 8-bit", "adapts per clinic"]),
        "blk_xray_secondary": (["Secondary"], ["characteristics", "contrast, texture,", "focus region"]),
        "blk_xray_features": (["Feature classifier"], ["logistic regression"]),
        "blk_xray_fusion": (["Late fusion"], ["weighted votes"]),
        "blk_decision": (["One decision per visit"], ["maximum expected utility"]),
        "blk_report": (["Screening report"], ["refer or not refer", "both probabilities",
                                             "secondary", "characteristics"]),
        "refer_if": "refer if",
        "def_ph": "heart branch: probability of an abnormal heart sound",
        "def_px": "X-ray branch: probability of pneumonia",
        "def_cmiss": "cost of missing a patient who needs care",
        "def_crefer": "cost of an unnecessary referral",

        # node cards
        "model_heart": "8-bit CNN · 5 classes",
        "model_xray": "8-bit MobileNetV2 · 2 classes",
        "include": "Include in this visit",
        "excluded_note": "Not part of this visit: the decision ignores this card.",
        "src_reference": "Reference case",
        "src_upload": "Upload a file",
        "upload_hint_heart": "WAV at any sample rate, at least 1.5 s",
        "upload_hint_xray": "PNG or JPEG of any size; it is centre-cropped to 96 × 96",
        "upload_waiting": "Waiting for a file.",
        "upload_tag": "Uploaded file: not a reference case, so there is no expected answer",
        "withheld_warn": "Heart result withheld: {reason}",
        "reason_too_short": "the recording is shorter than 1.5 s",
        "reason_unreadable": "the file could not be read as a WAV recording",
        "reason_quality_check": "too much sound energy above 1 kHz (measured {hi}, limit {thr})",
        "analysis_failed": "{branch} analysis failed: {err}",
        "close_call": "Close call between {a} and {b}",
        "fused_line": "After late fusion: {p} probability of {what}",
        "what_heart": "an abnormal heart sound",
        "what_xray": "pneumonia",
        "votes_title": "Late fusion votes",
        "vote_cnn": "CNN",
        "vote_features": "Feature classifier",
        "vote_metadata": "Metadata classifier",
        "weight": "weight",
        "fusion_fallback": "Fitted fusion parameters are not available here; this result is the calibrated "
                           "CNN vote alone.",
        "model_size": "8-bit model",
        "input_heart": "40 × 192 log-mel, 3 s windows",
        "input_xray": "96 × 96 grayscale",
        "media_input_xray": "What the model saw (96 × 96)",
        "clahe_label": "Contrast-enhanced view (CLAHE)",
        "occlusion_label": "Focus region: probability drop when each ninth is hidden",

        # patient details
        "patient_title": "Patient details",
        "patient_note": "Optional. Leave a field empty when it was not recorded; nothing is filled in by guess.",
        "age": "Age (years)",
        "sex": "Sex",
        "sex_female": "Female",
        "sex_male": "Male",
        "height": "Height (cm)",
        "weight_kg": "Weight (kg)",
        "pregnant": "Pregnant",
        "symptoms": "Reported symptoms",
        "symp_breathlessness": "Breathlessness",
        "symp_chest_pain": "Chest pain",
        "symp_cough": "Cough",
        "symp_fever": "Fever",
        "not_recorded": "Not recorded",
        "meta_none": "No patient details: the metadata vote is left out of both branches.",
        "meta_some": "{n} detail(s) recorded: the metadata vote is used (weight at most 0.3).",

        # visit decision
        "decision_title": "One decision per visit",
        "decision_hint": "p = 1 − (1 − pₕ)(1 − pₓ), compared with the clinic's threshold",
        "analyse_visit": "Analyse visit",
        "cost_ratio": "Cost ratio C_miss / C_refer",
        "cost_ratio_help": "How much worse a missed patient is than an unnecessary referral. "
                           "A higher ratio lowers the threshold and refers more patients.",
        "threshold_line": "Threshold τ = 1 / (1 + ratio) = {tau}",
        "decision_refer": "Refer for clinical review",
        "decision_refer_desc": "The probability of any finding, {p_any}, is at or above the clinic's threshold "
                               "of {tau} (cost ratio {ratio}:1). Refer the patient for review.",
        "decision_routine": "Routine follow-up",
        "decision_routine_desc": "The probability of any finding, {p_any}, is below the clinic's threshold "
                                 "of {tau} (cost ratio {ratio}:1). Continue routine community follow-up.",
        "no_usable_test": "No visit decision: no test gave a usable result. See the reason in the card, "
                          "then record or capture again.",
        "no_analysis_yet": "No visit analysed yet. Choose the inputs, then press Analyse visit.",
        "no_tests_included": "Include at least one test (heart sound or chest X-ray) in this visit.",
        "stale_note": "The inputs or patient details changed after this result. "
                      "Press Analyse visit to update it.",
        "lbl_heart": "Heart, pₕ",
        "lbl_xray": "X-ray, pₓ",
        "lbl_any": "Any finding, p",
        "lbl_tau": "Threshold, τ",
        "not_in_visit": "not in this visit",
        "withheld_short": "withheld",
        "not_analysed": "not analysed",
        "export_visit": "Export visit summary (JSON)",

        # batch results, chart and log
        "acc_cases": "{n} of {total} reference cases analysed",
        "ref_heart": "8-bit model on held-out fold 0: {acc} (Yaseen). 5-fold cross-validation: {cv}.",
        "ref_xray": "8-bit model on held-out fold 0: {acc} per image, {pacc} per patient (Kermany).",
        "batch_running": "Running the screening batch on {n} reference cases",
        "legend_miss": "differs from expected",
        "th_time": "Time (UTC)",
        "log_empty": "No analyses yet.",
        "batch_tag": "batch",

        # case details
        "sec_fusion": "Late fusion",
        "quality_pass": "Passed: energy above 1 kHz {hi} (limit {thr})",
        "quality_fail": "Failed: energy above 1 kHz {hi} (limit {thr})",
        "windows": "Windows analysed",
        "temperature": "Calibration temperature",
        "no_case_result": "Analyse this case first.",
        "d_bilateral": "both sides",
        "d_patient_right": "patient's right",
        "d_patient_left": "patient's left",
        "z_upper": "upper",
        "z_middle": "middle",
        "z_lower": "lower",
        "p_focal": "focal",
        "p_diffuse": "diffuse",
        "p_none": "none",
        "side_0": "viewer's left",
        "side_1": "centre",
        "side_2": "viewer's right",
        "region": "{zone} zone, {side}",
        "limits_normal_heart": "A normal screen does not exclude heart disease. Only the recorded sound, "
                               "from one chest position, is analysed.",
        # Chiron strings restated: no em dash, and an accuracy label that fits both branches
        "generic_note": "General reference for this finding, not derived from this recording",
        "cv_note": "Reference accuracy",
    },
    "hi": {
        "demo_label": "सार्वजनिक डेमो",
        "public_demo_note": "केवल दिए गए संदर्भ मामलों या अन्य सार्वजनिक डेटा का उपयोग करें। किसी वास्तविक मरीज़ की "
                            "रिकॉर्डिंग या एक्स-रे कभी अपलोड न करें। परिणाम केवल प्रदर्शन के लिए हैं, निदान नहीं।",
        "model_ready": "{name} मॉडल तैयार",
        "model_missing": "{name} मॉडल उपलब्ध नहीं",
        "dataset_heart": "Yaseen 2018 हृदय ध्वनियाँ, 5 वर्ग; संदर्भ मामले होल्ड-आउट फ़ोल्ड 0 से",
        "dataset_xray": "Kermany 2018 बाल चिकित्सा छाती एक्स-रे; संदर्भ मामले होल्ड-आउट फ़ोल्ड 0 से",
        "footer_repo": "स्रोत: github.com/AI-Projects-S1-26-27/T4_CardioNary",
        "retention": "रिकॉर्ड केवल इसी ब्राउज़र सत्र में रहते हैं और सत्र समाप्त होने पर हट जाते हैं। "
                     "सर्वर पर कुछ भी संग्रहीत नहीं होता।",

        "arch_title": "प्रणाली संरचना",
        "arch_hint": "हाइलाइट किए गए ब्लॉक आपकी वर्तमान गतिविधि दिखाते हैं",
        "cap_start": "यहाँ से शुरू करें: नीचे हर कार्ड में एक संदर्भ मामला चुनें या फ़ाइल अपलोड करें। "
                     "रोगी विवरण वैकल्पिक हैं।",
        "cap_heart": "हृदय ध्वनि चुनी गई। दूसरे मामले पर जाने से उसका विश्लेषण होता है।",
        "cap_xray": "छाती का एक्स-रे चुना गया। दूसरे मामले पर जाने से उसका विश्लेषण होता है।",
        "cap_wait_upload": "{branch} कार्ड में अपलोड की प्रतीक्षा है।",
        "cap_meta": "रोगी विवरण दोनों शाखाओं के मेटाडेटा वोट में जाते हैं। इनका उपयोग करने के लिए "
                    "'जाँच का विश्लेषण करें' दबाएँ।",
        "cap_decision": "शामिल जाँचें और लागत अनुपात जाँच का निर्णय तय करते हैं।",
        "cap_running": "चल रहा है: {stage}",
        "cap_done": "विश्लेषण पूरा हुआ। जाँच का निर्णय और स्क्रीनिंग रिपोर्ट नीचे हैं।",
        "cap_branch_done": "{branch} परिणाम तैयार। संयुक्त निर्णय के लिए 'जाँच का विश्लेषण करें' दबाएँ।",
        "cap_batch": "स्क्रीनिंग बैच: {branch} मामला {i} / {n}",
        "cap_batch_done": "स्क्रीनिंग बैच पूरा हुआ: नीचे बैच परिणाम और केस लॉग देखें।",
        "lg_waiting": "इनपुट की प्रतीक्षा",
        "lg_ready": "तैयार",
        "lg_active": "अभी चल रहा है",
        "lg_done": "पूरा",
        "lg_withheld": "रोका गया",
        "lg_unused": "उपयोग नहीं हुआ",
        "lg_focus": "आप यहाँ हैं",
        "lg_cnn": "कॉम्पैक्ट CNN (8-बिट; हर नए क्लिनिक के अनुसार ढलता है)",
        "lg_classic": "पारंपरिक प्रोसेसिंग, मेटाडेटा, लेट फ्यूज़न और निर्णय",

        "stg_heart_load_and_resample": "हृदय ध्वनि पढ़ना, 8 kHz पर बदलना",
        "stg_heart_quality_check": "गुणवत्ता जाँच (लंबाई, 1 kHz से ऊपर शोर)",
        "stg_heart_front_end": "स्पेक्ट्रोग्राम (40 × 192 लॉग-मेल)",
        "stg_heart_cnn_8bit": "8-बिट हृदय CNN",
        "stg_heart_secondary_characteristics": "द्वितीयक विशेषताएँ (हृदय गति, लय, ध्वनि का समय)",
        "stg_heart_systole_or_diastole": "सिस्टोल या डायस्टोल समय",
        "stg_heart_votes_and_fusion": "वोट और लेट फ्यूज़न (हृदय)",
        "stg_xray_load_image": "एक्स-रे को 96 × 96 ग्रेस्केल छवि के रूप में पढ़ना",
        "stg_xray_cnn_8bit": "8-बिट एक्स-रे CNN",
        "stg_xray_secondary_characteristics": "द्वितीयक विशेषताएँ (कंट्रास्ट, बनावट)",
        "stg_xray_occlusion_map": "फ़ोकस क्षेत्र (3 × 3 ऑक्लूज़न मानचित्र)",
        "stg_xray_votes_and_fusion": "वोट और लेट फ्यूज़न (एक्स-रे)",
        "stg_decision": "प्रति जाँच एक निर्णय",
        "stg_report": "स्क्रीनिंग रिपोर्ट",

        "blk_heart_audio": (["हृदय ध्वनि"], ["3 सेकंड"]),
        "blk_spectrogram": (["स्पेक्ट्रोग्राम"], ["40 × 192"]),
        "blk_heart_cnn": (["हृदय CNN"], ["कॉम्पैक्ट, 8-बिट", "क्लिनिक अनुसार ढलता है"]),
        "blk_heart_secondary": (["द्वितीयक"], ["विशेषताएँ", "हृदय गति, लय,", "ध्वनि का समय"]),
        "blk_heart_features": (["फ़ीचर क्लासिफ़ायर"], ["लॉजिस्टिक रिग्रेशन"]),
        "blk_heart_fusion": (["लेट फ्यूज़न"], ["भारित वोट"]),
        "blk_metadata": (["रोगी मेटाडेटा और लक्षण"], ["उम्र, लिंग, शरीर का आकार, लक्षण"]),
        "blk_meta_classifier": (["मेटाडेटा क्लासिफ़ायर"], ["लॉजिस्टिक रिग्रेशन"]),
        "blk_xray": (["छाती का", "एक्स-रे"], ["कोई भी आकार"]),
        "blk_grayscale": (["ग्रेस्केल छवि"], ["96 × 96"]),
        "blk_xray_cnn": (["एक्स-रे CNN"], ["MobileNetV2, 8-बिट", "क्लिनिक अनुसार ढलता है"]),
        "blk_xray_secondary": (["द्वितीयक"], ["विशेषताएँ", "कंट्रास्ट, बनावट,", "फ़ोकस क्षेत्र"]),
        "blk_xray_features": (["फ़ीचर क्लासिफ़ायर"], ["लॉजिस्टिक रिग्रेशन"]),
        "blk_xray_fusion": (["लेट फ्यूज़न"], ["भारित वोट"]),
        "blk_decision": (["प्रति जाँच एक निर्णय"], ["अधिकतम अपेक्षित उपयोगिता"]),
        "blk_report": (["स्क्रीनिंग रिपोर्ट"], ["रेफर करें या नहीं", "दोनों संभावनाएँ", "द्वितीयक", "विशेषताएँ"]),
        "refer_if": "रेफर करें यदि",
        "def_ph": "हृदय शाखा: असामान्य हृदय ध्वनि की संभावना",
        "def_px": "एक्स-रे शाखा: निमोनिया की संभावना",
        "def_cmiss": "देखभाल की ज़रूरत वाले मरीज़ को छोड़ देने की लागत",
        "def_crefer": "अनावश्यक रेफरल की लागत",

        "model_heart": "8-बिट CNN · 5 वर्ग",
        "model_xray": "8-बिट MobileNetV2 · 2 वर्ग",
        "include": "इस जाँच में शामिल करें",
        "excluded_note": "इस जाँच का हिस्सा नहीं: निर्णय इस कार्ड को नहीं गिनता।",
        "src_reference": "संदर्भ मामला",
        "src_upload": "फ़ाइल अपलोड करें",
        "upload_hint_heart": "किसी भी सैंपल दर पर WAV, कम से कम 1.5 सेकंड",
        "upload_hint_xray": "किसी भी आकार की PNG या JPEG; बीच से 96 × 96 में काटी जाती है",
        "upload_waiting": "फ़ाइल की प्रतीक्षा है।",
        "upload_tag": "अपलोड की गई फ़ाइल: संदर्भ मामला नहीं, इसलिए कोई अपेक्षित उत्तर नहीं",
        "withheld_warn": "हृदय परिणाम रोका गया: {reason}",
        "reason_too_short": "रिकॉर्डिंग 1.5 सेकंड से छोटी है",
        "reason_unreadable": "फ़ाइल को WAV रिकॉर्डिंग के रूप में पढ़ा नहीं जा सका",
        "reason_quality_check": "1 kHz से ऊपर बहुत अधिक ध्वनि ऊर्जा (मापी गई {hi}, सीमा {thr})",
        "analysis_failed": "{branch} विश्लेषण विफल: {err}",
        "close_call": "{a} और {b} के बीच बहुत कम अंतर",
        "fused_line": "लेट फ्यूज़न के बाद: {what} की संभावना {p}",
        "what_heart": "असामान्य हृदय ध्वनि",
        "what_xray": "निमोनिया",
        "votes_title": "लेट फ्यूज़न वोट",
        "vote_cnn": "CNN",
        "vote_features": "फ़ीचर क्लासिफ़ायर",
        "vote_metadata": "मेटाडेटा क्लासिफ़ायर",
        "weight": "भार",
        "fusion_fallback": "फ्यूज़न के प्रशिक्षित पैरामीटर यहाँ उपलब्ध नहीं हैं; यह परिणाम केवल कैलिब्रेटेड CNN वोट है।",
        "model_size": "8-बिट मॉडल",
        "input_heart": "40 × 192 लॉग-मेल, 3 सेकंड विंडो",
        "input_xray": "96 × 96 ग्रेस्केल",
        "media_input_xray": "मॉडल ने जो देखा (96 × 96)",
        "clahe_label": "कंट्रास्ट-संवर्धित दृश्य (CLAHE)",
        "occlusion_label": "फ़ोकस क्षेत्र: हर नौवाँ भाग छिपाने पर संभावना में गिरावट",

        "patient_title": "रोगी विवरण",
        "patient_note": "वैकल्पिक। जो जानकारी दर्ज नहीं हुई, उसे खाली छोड़ें; कुछ भी अनुमान से नहीं भरा जाता।",
        "age": "उम्र (वर्ष)",
        "sex": "लिंग",
        "sex_female": "महिला",
        "sex_male": "पुरुष",
        "height": "ऊंचाई (सेमी)",
        "weight_kg": "वजन (किग्रा)",
        "pregnant": "गर्भवती",
        "symptoms": "दर्ज लक्षण",
        "symp_breathlessness": "सांस फूलना",
        "symp_chest_pain": "छाती में दर्द",
        "symp_cough": "खांसी",
        "symp_fever": "बुखार",
        "not_recorded": "दर्ज नहीं",
        "meta_none": "कोई रोगी विवरण नहीं: दोनों शाखाओं में मेटाडेटा वोट शामिल नहीं है।",
        "meta_some": "{n} विवरण दर्ज: मेटाडेटा वोट का उपयोग होता है (भार अधिकतम 0.3)।",

        "decision_title": "प्रति जाँच एक निर्णय",
        "decision_hint": "p = 1 − (1 − pₕ)(1 − pₓ), क्लिनिक की सीमा से तुलना",
        "analyse_visit": "जाँच का विश्लेषण करें",
        "cost_ratio": "लागत अनुपात C_miss / C_refer",
        "cost_ratio_help": "छूटा हुआ मरीज़ अनावश्यक रेफरल से कितना अधिक नुकसानदेह है। अनुपात बढ़ाने से सीमा घटती है "
                           "और अधिक मरीज़ रेफर होते हैं।",
        "threshold_line": "सीमा τ = 1 / (1 + अनुपात) = {tau}",
        "decision_refer": "क्लिनिकल समीक्षा के लिए रेफर करें",
        "decision_refer_desc": "किसी भी निष्कर्ष की संभावना {p_any}, क्लिनिक की सीमा {tau} के बराबर या उससे अधिक है "
                               "(लागत अनुपात {ratio}:1)। रोगी को समीक्षा के लिए रेफर करें।",
        "decision_routine": "नियमित निगरानी",
        "decision_routine_desc": "किसी भी निष्कर्ष की संभावना {p_any}, क्लिनिक की सीमा {tau} से कम है "
                                 "(लागत अनुपात {ratio}:1)। नियमित सामुदायिक निगरानी जारी रखें।",
        "no_usable_test": "कोई निर्णय नहीं: किसी भी जाँच से उपयोगी परिणाम नहीं मिला। कार्ड में कारण देखें, "
                          "फिर दोबारा रिकॉर्ड करें।",
        "no_analysis_yet": "अभी कोई जाँच विश्लेषित नहीं हुई। इनपुट चुनें, फिर 'जाँच का विश्लेषण करें' दबाएँ।",
        "no_tests_included": "इस जाँच में कम से कम एक परीक्षण (हृदय ध्वनि या छाती का एक्स-रे) शामिल करें।",
        "stale_note": "इस परिणाम के बाद इनपुट या रोगी विवरण बदले गए। अपडेट करने के लिए 'जाँच का विश्लेषण करें' दबाएँ।",
        "lbl_heart": "हृदय, pₕ",
        "lbl_xray": "एक्स-रे, pₓ",
        "lbl_any": "कोई भी निष्कर्ष, p",
        "lbl_tau": "सीमा, τ",
        "not_in_visit": "इस जाँच में नहीं",
        "withheld_short": "रोका गया",
        "not_analysed": "विश्लेषण नहीं हुआ",
        "export_visit": "जाँच सारांश निर्यात करें (JSON)",

        "acc_cases": "{total} में से {n} संदर्भ मामले विश्लेषित",
        "ref_heart": "होल्ड-आउट फ़ोल्ड 0 पर 8-बिट मॉडल: {acc} (Yaseen)। 5-फ़ोल्ड क्रॉस-वैलिडेशन: {cv}।",
        "ref_xray": "होल्ड-आउट फ़ोल्ड 0 पर 8-बिट मॉडल: प्रति छवि {acc}, प्रति रोगी {pacc} (Kermany)।",
        "batch_running": "{n} संदर्भ मामलों पर स्क्रीनिंग बैच चल रहा है",
        "legend_miss": "अपेक्षित से भिन्न",
        "th_time": "समय (UTC)",
        "log_empty": "अभी कोई विश्लेषण नहीं।",
        "batch_tag": "बैच",

        "sec_fusion": "लेट फ्यूज़न",
        "quality_pass": "उत्तीर्ण: 1 kHz से ऊपर ऊर्जा {hi} (सीमा {thr})",
        "quality_fail": "अनुत्तीर्ण: 1 kHz से ऊपर ऊर्जा {hi} (सीमा {thr})",
        "windows": "विश्लेषित विंडो",
        "temperature": "कैलिब्रेशन तापमान",
        "no_case_result": "पहले इस मामले का विश्लेषण करें।",
        "d_bilateral": "दोनों ओर",
        "d_patient_right": "रोगी का दायाँ भाग",
        "d_patient_left": "रोगी का बायाँ भाग",
        "z_upper": "ऊपरी",
        "z_middle": "मध्य",
        "z_lower": "निचला",
        "p_focal": "केंद्रित",
        "p_diffuse": "फैला हुआ",
        "p_none": "कोई नहीं",
        "side_0": "दर्शक का बायाँ",
        "side_1": "मध्य",
        "side_2": "दर्शक का दायाँ",
        "region": "{zone} क्षेत्र, {side}",
        "limits_normal_heart": "सामान्य परिणाम हृदय रोग को खारिज नहीं करता। केवल छाती की एक स्थिति से रिकॉर्ड की गई "
                               "ध्वनि का ही विश्लेषण होता है।",
        "generic_note": "इस निष्कर्ष के लिए सामान्य संदर्भ, इस रिकॉर्डिंग से प्राप्त नहीं",
        "cv_note": "संदर्भ सटीकता",
    },
}


@lru_cache(maxsize=1)
def chiron() -> dict:
    """The Chiron station's interface and clinical text, as exported."""
    return json.loads(CHIRON_FILE.read_text(encoding="utf-8"))


def tr(lang: str, key: str, **fmt) -> str:
    """One interface string: CardioNary's own first, then Chiron's, then English."""
    lang = lang if lang in LANGS else "en"
    for source in (APP[lang], chiron()[lang]["ui"], APP["en"], chiron()["en"]["ui"]):
        if key in source:
            value = source[key]
            return value.format(**fmt) if fmt and isinstance(value, str) else value
    return key


def clinical(lang: str, modality: str, cls: str) -> dict | None:
    """Chiron's clinical reference for one class: name, finding, urgency, actions, confirm, associations, limits."""
    lang = lang if lang in LANGS else "en"
    try:
        idx = CLASSES[modality].index(cls)
    except (KeyError, ValueError):
        return None
    entry = chiron()[lang]["clinical"].get(CHIRON_NODE[modality], {}).get(str(idx))
    if entry is None:
        return None
    entry = dict(entry)
    if modality == "heart" and cls == "Normal":
        # Chiron analysed exactly 3 s; CardioNary analyses every window of the recording.
        entry["limits"] = tr(lang, "limits_normal_heart")
    return entry


def class_name(lang: str, modality: str, cls: str | None) -> str:
    """The clinical name of a class, e.g. "AS" -> "Aortic stenosis"; the short code if unknown."""
    if cls is None:
        return "-"
    entry = clinical(lang, modality, cls)
    return entry["name"] if entry else cls
