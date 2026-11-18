# Vedansh — Method and S6 results

## Method (one-page contribution)

This work covers the chest X-ray branch of CardioNary, an offline screening-support interface. The model is a MobileNetV2 CNN trained on pediatric Kermany chest radiographs to classify Normal versus Pneumonia. Images are converted to grayscale, center-cropped to a square, resized to 96 × 96 pixels, and normalized with the mean and standard deviation fitted on Kermany training data. The five-fold protocol keeps patients disjoint; fold 0 is held out when producing the final model. RSNA is treated as an unseen adult cohort and is used only for evaluation, not for fitting the CNN or its image normalization.

The deployed X-ray inference uses the fully integer INT8 TFLite model. The CNN probability is temperature-scaled with the frozen value fitted from Kermany out-of-fold predictions. The dashboard then reads the committed fusion parameters; it does not fit a model or contact a server at prediction time. The current weights are 0.85 for the CNN, 0.00 for the image-feature vote, and 0.15 for the age-and-sex metadata vote. Missing optional inputs cause the available vote weights to be renormalized. The metadata vote is capped at 0.30 and excludes acquisition details such as view position. Symptoms are collected by the shared dashboard for visit context, but are not inputs to this X-ray metadata classifier.

For interpretability, the X-ray path also calculates a fixed set of image characteristics and a CLAHE-enhanced preview. Its 3 × 3 occlusion map masks each image region in turn and reports the change in calibrated CNN probability. These are visual and descriptive aids, not independent diagnoses. The page shows the calibrated Normal/Pneumonia probabilities, available vote probabilities and weights, image characteristics, the CLAHE preview, and the occlusion map. The shared Streamlit interface supports English and Hindi and is configured with the roadmap's blue accent. Model files, normalization values, calibration, and fusion parameters are stored locally, so one image can be evaluated without internet access.

The evaluation separates development from external testing. Model and feature-vote choices use Kermany's patient-disjoint folds; the RSNA evaluation contains one image per patient and uses Kermany normalization. Bootstrap intervals use 1,000 resamples with seed 0. Test-time adaptation uses unlabeled images only, and every patient selected for adaptation is excluded from its score set. Synthetic corruption tests use held-out Kermany fold 0. Full protocol definitions and generated outputs are in `lung/eval/`, `reports/xray_rsna.json`, `reports/xray_fusion.json`, and `reports/xray_robustness.json`.

## S6 results, limitations, and tool

### Unseen RSNA cohort

The final model was evaluated on 14,860 RSNA patients using Kermany training normalization. The float model reached AUROC 0.9071 (95% bootstrap CI 0.9017–0.9121), accuracy 0.8458, sensitivity 0.7751 at 90% specificity, and ECE 0.0684. INT8 AUROC was 0.9022 (0.8969–0.9071), accuracy 0.8455, and sensitivity 0.7660 at 90% specificity. Float/INT8 class agreement on RSNA was 96.10% (95% CI 95.79–96.42%). The Kermany-trained image-feature vote reduced RSNA AUROC by 0.0084 (95% CI −0.0099 to −0.0070). Adding age/sex increased AUROC by 0.0140 (95% CI 0.0119–0.0163), below the roadmap's +0.02 target. These results indicate domain shift from pediatric training images to adult RSNA images; the features and metadata did not meet the planned improvement targets.

### Robustness and adaptation

On held-out Kermany fold 0, 32 unlabeled images were used for adaptation and excluded from scoring; the remaining scores cover 1,130 images from 527 patients. The full plain/adapted accuracy and AUROC results for all four corruption types at all five severities, with patient-bootstrap intervals, are in `reports/xray_robustness.json`. Adaptation helped at stronger blur, noise, and contrast corruption levels, but did not consistently help at lower severity and reduced performance under stronger JPEG compression. This synthetic-shift result does not transfer to RSNA: the RSNA adaptation comparison in `reports/xray_fusion.json` showed lower AUROC than the plain CNN. The plain model is therefore retained for the dashboard.

### Offline tool and limits

The X-ray page is part of `app/app.py` (the current repository does not have a separate `lung/app_xray.py`). It accepts a PNG/JPG/JPEG upload plus age and sex, and provides symptom checkboxes. It displays the two calibrated class probabilities, current vote weights, characteristic table, CLAHE view, and 3 × 3 occlusion map in English or Hindi. The local end-to-end X-ray pipeline measured 2.646 ms per case in the committed edge report; the INT8 model is 625,080 bytes. Original INT8 agreement with float on Kermany fold 0 was 98.95% at item level (95% CI 98.39–99.52%) and 98.39% at patient level (97.31–99.29%). The separate adapted INT8 model's agreement is below the 98% target, so it is not used in this tool. Performance metrics are research evaluations, not evidence of clinical effectiveness; validation on representative clinical sites and prospective data remains necessary.

## Individual contribution

Vedansh maintained the X-ray RSNA age parsing fix, rebuilt the RSNA manifest with DICOM ages, reran the RSNA evaluation and robustness sweep, verified the offline X-ray inference path and contributed this method/results write-up. The implementation and outputs are recorded in the code and JSON files cited above.

## AI-use note

OpenAI Codex was used to inspect repository code and report JSON, identify the missing accuracy bootstrap intervals, draft the evaluation change and report text, and help run the project commands. The contributor reviewed the changed code, checked the reported values against the generated JSON files, and ran the focused metric tests, RSNA evaluation, corruption sweep, and project test suite. Any AI-assisted text or code should be reviewed by the team before publication.
