# CardioNary Report Sections

## Results and Evaluation

CardioNary was evaluated through separate experiments for the heart-sound and chest X-ray pipelines. The evaluation focused on model performance, generalisation to new datasets, the usefulness of additional features and patient information, reliability under different input conditions, and suitability for deployment on low-power devices.

**Model Performance and Generalisation:** The heart-sound model achieved 99.3% accuracy on its original evaluation dataset. However, its performance dropped significantly when tested on the independent PhysioNet 2022 dataset, achieving an AUROC of only 0.52, which is close to random prediction. In comparison, the chest X-ray model achieved 97.3% accuracy at the patient level and an AUROC of 0.907 on the adult RSNA dataset, despite initially being trained on pediatric X-ray images. This showed better transfer to a different dataset in the X-ray pipeline.

**Additional Features and Patient Information:** We evaluated whether additional medical features and patient information could improve the main model predictions. The feature-based classifiers did not provide meaningful improvements in either pipeline. For the heart-sound model, patient metadata improved the AUROC by 0.071, meeting the target improvement of 0.02. For the X-ray model, the improvement was only 0.005. Missing age values caused by a data-processing error affected the evaluation of the X-ray metadata classifier.

**Calibration and Adaptation:** Calibration improved the reliability of the predicted probabilities. For the heart model, the calibration error decreased from 0.058 to 0.006, while for the X-ray model it decreased from 0.027 to 0.025. We also tested adaptation under changing input conditions. For heart recordings with 20 dB white noise, accuracy improved from 42% to 96%. On PhysioNet, the referral AUROC increased from 0.52 to 0.70, although macro-F1 at the 0.5 threshold remained unchanged. Adaptation also reduced performance on clean Yaseen recordings.

**Model Compression and Deployment:** Both models were converted into smaller 8-bit versions to reduce storage and computational requirements. The heart model met the required targets for model size, speed, and agreement with its original version. The X-ray model met the size and speed requirements but achieved only 80% prediction agreement, falling short of the intended 98% target. Overall, the experiments showed promising results for the X-ray pipeline, while the heart-sound pipeline requires further improvement in generalisation.

## Discussion and Limitations

The results of CardioNary highlight both the potential and the challenges of developing a lightweight screening system that combines heart-sound and chest X-ray analysis. Although both pipelines performed well on their original datasets, their ability to handle completely new data was different.

The chest X-ray model showed encouraging results when transferred from pediatric training data to adult RSNA images, achieving an AUROC of 0.907. This suggests that the model learned useful image patterns that could also be applied to a different age group. However, the feature-based classifier did not improve performance significantly. Additionally, an error in the age-processing function caused all 14,863 RSNA age values to be lost because the parser expected age values with a unit letter, while the dataset stored plain numbers. This affected the reliability of the metadata evaluation.

The heart-sound pipeline faced greater challenges. Despite achieving high accuracy on its original dataset, its AUROC dropped to 0.52 on PhysioNet 2022. The model also classified a large proportion of recordings as abnormal, indicating that it struggled to distinguish between normal and abnormal sounds from a different source. Furthermore, the sound-quality check rejected approximately 91% of PhysioNet recordings and 82% of recordings from the clinic, making it too restrictive for practical use with unfamiliar recording devices.

Another limitation was that adaptation, although helpful under noisy conditions, did not consistently improve performance. It improved the referral AUROC on PhysioNet but did not improve macro-F1 at the selected decision threshold. It also reduced performance on clean training-domain recordings. Similarly, the feature-based classifiers did not provide the expected improvements in either pipeline.

There were also limitations in model compression. While the heart model met the deployment requirements, the X-ray 8-bit model available on the main branch did not match the original floating-point model, resulting in only 80% prediction agreement instead of the required 98%. This indicates the need to verify the exported model and its conversion process.

Finally, evaluation on the BMD-HS dataset was not completed, and the system has not yet been established as a clinically validated diagnostic tool. Its performance across larger and more diverse patient populations and recording devices requires further investigation.

Overall, CardioNary demonstrates the feasibility of combining two screening pipelines into a lightweight system. However, improving cross-dataset performance, reducing unnecessary rejections, fixing data-processing issues, and ensuring consistent model outputs are essential steps before practical clinical use.

## AI Usage Details

AI tools, including ChatGPT and OpenAI Codex, were used during the development of CardioNary to assist with understanding the existing codebase, debugging, exploring implementation approaches, and improving technical documentation. They also helped in reviewing code structure, understanding model evaluation procedures, and organising parts of the project report.

AI-generated suggestions were treated as assistance rather than final solutions. The team reviewed the suggested code and explanations, made necessary corrections, and verified implementations through actual execution and testing. Model training, dataset preparation, experiments, performance measurements, and final evaluation were based on the project's implementation and experimental results.

AI tools were not treated as a replacement for experimental verification. The final technical decisions, interpretation of results, and conclusions were reviewed by the project team.
