Reviewer 2 Response and Revision Draft

1. Problem Statement
Status: Good (no change requested).

2. Literature Review
Status: Good (no change requested).

3. Research Methodology
Status: Good (minor additions below to improve precision).

4. Results and Conclusions
Status: Good (minor additions below to strengthen quantitative reporting).

5. General Evaluation of the Paper
Status: Good.

6. Problem Type(s) Addressed
Educational; Industrial/Organizational; Entrepreneurial/Innovation.

7. Decision of Reviewer
Accepted with Minor Modifications.

8. Reviewer Comments and Revisions

A. Clarify Novelty and Contributions
Add the following paragraph near the end of the Introduction:

"This work focuses on real-time, on-device combat analytics using lightweight pose estimation and sequence modeling. The approach targets practical training environments by emphasizing low-latency inference, robustness to single-camera constraints, and interpretable tactical feedback."

Add the following bullet list in the Introduction under a heading such as "Contributions":
- On-device, real-time pose-based action analysis for boxing/MMA scenarios.
- Optimized MediaPipe pose estimation pipeline tailored to combat motion and single-angle video.
- Sequence-based next-move prediction using a hybrid CNN-LSTM approach (or LSTM-based temporal modeling) for tactical forecasting.
- Practical feedback outputs for training (action distribution, tempo, transition probabilities).

B. Strengthen Dataset and Evaluation Description
Add a Dataset subsection with explicit counts and duration. Example text (replace numbers with your final values):

"The dataset consists of N clips captured from single-angle video, totaling T minutes. Two fighters are tracked per clip. The labeled action set includes guard_position, punch_left, punch_right, forward_movement, and backward_movement (boxing-only variant). The dataset contains E events with per-event features (hand_speed, leg_speed, movement_intensity)."

Add an Evaluation subsection with metrics. Example text:

"We report accuracy and macro-F1 for multi-class next-move prediction. On the boxing-only dataset, the best baseline (class-weighted Logistic Regression, window size 3) achieves 0.444 accuracy and 0.489 macro-F1, outperforming Markov and Most-Frequent baselines. These results indicate that short-term sequences capture tactical cues, though class imbalance and small sample size limit absolute performance."

If you have latency/FPS numbers, add:

"On-device inference runs at approximately X FPS with end-to-end latency of Y ms per frame using the MediaPipe pose estimator." 

C. Improve Methodological Precision
Add a brief justification paragraph in Methodology:

"MediaPipe Pose is chosen for its low-latency, mobile-ready inference and reliable 2D landmark detection under single-camera conditions. CNN layers capture short-term motion patterns from engineered features, while LSTM layers model temporal dependencies for next-move prediction. The pipeline is optimized for on-device inference by using lightweight pose estimation and small sequence windows (3-15 events), reducing compute and memory overhead."

Add a short pseudocode or flowchart reference, for example:

Algorithm 1: Real-Time Pose-to-Action Pipeline
1. Read frame t from video stream
2. Run MediaPipe pose estimation to obtain landmarks
3. Compute features (speed, intensity, temporal window aggregates)
4. Predict current action and next action
5. Update tactical statistics and feedback overlay
6. Repeat for next frame

D. Enhance Results and Comparative Analysis
Add a comparison table in Results (example structure):

Table: Next-Move Prediction Performance (boxing-only)
- Most Frequent: accuracy 0.389, macro-F1 0.112
- Markov Baseline: accuracy 0.444, macro-F1 0.206
- Logistic Regression (balanced): accuracy 0.444, macro-F1 0.489
- Random Forest (balanced): accuracy 0.278, macro-F1 0.087

Add one or two plots from the notebook:
- Action distribution bar chart
- Action transition heatmap

E. Refine Technical Writing and Structure
Guidance applied:
- Use consistent terms: MediaPipe, pose estimation, real-time feedback.
- Replace informal phrases with concise IEEE/Springer tone.
- Avoid repetition in methodology and results.

F. Expand Limitations and Future Scope
Add a Limitations and Future Work paragraph:

"The current dataset is limited in size and diversity, which constrains generalization and rare-action detection. Single-angle capture introduces occlusions and viewpoint sensitivity, especially during close-range exchanges. Future work will incorporate multi-camera capture and view-fusion to improve pose accuracy, as well as 3D pose reconstruction to better model strike trajectories and timing. We also plan to extend the dataset across more fighters, longer sessions, and broader martial-arts styles to improve robustness."

Notes for Integration
- Place the Contributions list in the Introduction.
- Place Dataset and Evaluation subsections under Methodology or Experiments.
- Place the baseline comparison table under Results.
- Place Limitations/Future Work at the end of Conclusions.
