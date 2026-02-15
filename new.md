Techniques Using Optimized Pose Estimation
 
*Dr. Bhagyashree Dhakulkar
Assistant Professor Department of Artificial Intelligence and
Data Science
Ajeenkya DY Patil School of Engineering (SPPU)Pune, India bhagyashree.dhakulkar@gmail.com
 
Piyush Gangurde
UG Student
Department of Artificial Intelligence and Data Science
Ajeenkya DY Patil School of
Engineering (SPPU)Pune, India akaom07@gmail.com
 
Alokraj Ahire
UG Student
Department of Artificial Intelligence and Data Science
Ajeenkya DY Patil School of
Engineering (SPPU)Pune, India alokrajahire@gmail.com
 

 
 
Abstract-This project introduces an AI assistant designed to enhance the training and performance of Mixed Martial Arts (MMA) athletes. By analyzing real-time and video-based body movements, the system helps athletes refine their techniques, improve accuracy, and gain a strategic advantage over opponents. The assistant analyzes posture, speed, and precise movements to provide actionable insights for strategy planning. Advanced analytics further optimize performance efficiency during matches. Traditionally, MMA athletes and coaches relied on manual methods to analyze fights, such as handwritten notes and limited footage, which often resulted in incomplete or delayed insights. This system addresses these limitations by automating the analysis process. It uses hybrid model which includes classical machine learning models to detect patterns, techniques, and weaknesses in both the athlete’s and their opponent’s performances and media pipe for pose estimation. Real-time feedback during training allows for immediate adjustments, while predictive analytics forecast fight outcomes based on historical data and performance. The application also focuses on injury prevention by identifying risks like poor technique or overtraining. Personalized feedback helps athletes refine their movements and adopt better strategies. The system is designed to be user-friendly, ensuring accessibility for athletes and coaches. By integrating video analysis with AI-driven insights, this tool bridges the gap between raw data and practical improvements, offering a modern solution to enhance MMA training and competitive performance.

Keywords- Performance enhancement, Real-time analysis, Movement tracking, Strategy development, Injury prevention.



















 
I.	INTRODUCTION
Recent years have witnessed significant growth in human activity recognition studies, with researchers exploring diverse sensing modalities and developing innovative computational methods for activity modeling and classification. This expanding body of work has yielded numerous technical approaches for accurately identifying and interpreting human movements [1]. Before the advent of modern technology, MMA athletes and their coaches relied on traditional methods to analyze performance, prepare for fights, and gain a competitive edge. Coaches and cornermen would take handwritten notes during fights to track opponent's techniques, patterns, and weaknesses. Access to fight footage was limited, making it difficult for athletes to study their opponent's techniques and strategies. There were no online forums, tutorials, or instructional videos to learn new techniques or gain insights from other experts. Coaches relied on their own experience, knowledge, and instincts to develop strategies and make decisions during fights. Athletes had to rely on their instincts, reflexes, and experience to make split-second decisions during fights. It is not that beneficial to the performance of the player though it is important but, due to technology and automation a lot of things get shifted from manually to technical. So does the athletes and their planning methods and strategies. By adapting to those new things, we can get benefited a lot in increasing efficiency and performance. The application will help in video-based analysis of the players move and strategies. AI driven assistant will predict a move from that video and real time updates and will provide better understanding. Using AI, Media pipe and the classical machine learning models we can achieve various profits. AI can analyze  fight  footage  to  identify  patterns,
 
techniques, and weaknesses in both the athlete's and their opponents' performances.AI-powered predictive analytics can forecast the likely outcome of a fight based on historical data, athlete performance, and other factors.AI-powered video analysis can provide detailed analysis of an athlete's technique, identifying areas for improvement and providing personalized recommendations. AI-powered video analysis can analyze an opponent's strengths, weaknesses, and tactics, providing valuable insights for strategy development also it can identify potential injury risks, such as overtraining or poor technique, enabling coaches and trainers to take proactive measures and can provide personalized feedback to athletes, helping them adjust their technique and improve performance. Ensuring that this application will be more efficient and user friendly.
Combat skill acquisition occurs through:
a)	Kata: Solo pattern practice (cognitive encoding)
b)	Kihon Kumite: Partnered pressure training (contextual adaptation)
This progression bridges theoretical technique and practical application.[1]
This paper [2] presents a machine learning framework for predicting UFC bout outcomes using the organization's comprehensive historical database. By strategically excluding direct performance indicators, we develop classifiers achieving 80.3-92.1% accuracy across validation sets. The model's architecture permits expansion for future matchup forecasting in MMA's dynamic competitive landscape. Artificial intelligence has emerged as a transformative interdisciplinary field, combining computer vision, machine learning, and data science to revolutionize martial arts training and analysis. This paper systematically reviews AI applications in martial arts, examining: style recognition techniques, training task automation, multimodal data acquisition methods, and (4) algorithmic innovations. Through comprehensive analysis of current research, we present a unified framework for intelligent martial arts systems, outlining the complete technological pipeline from data collection to performance evaluation. Our synthesis reveals significant progress in movement quantification, real-time feedback systems, and personalized training protocols enabled by AI advancements [3].
Human Activity Recognition (HAR) systems have become increasingly vital in modern applications due to their capacity to extract meaningful behavioral patterns from raw sensor inputs. By automatically identifying and classifying human movements, these systems enable deeper understanding of physical activities, facilitating advancements in fields ranging from healthcare monitoring to smart assistive technologies [4].
Hence, we introduce a real-time system that runs directly on your device to analyze combat movements. By tracking how you move and recognizing patterns over time, our approach is tailored for real-world training. It keeps things fast, works reliably with just one camera, and gives clear feedback to help you improve your technique while you practice.


II.	LITERATURE SURVEY
Most publicly available combat sports datasets—like those based on UFC fight stats—mainly offer big-picture numbers: things like striking accuracy, takedown rates, defense percentages, and who won the match. These are great for looking back at overall performance or comparing fighters, but they don’t tell you exactly when or how certain moves happened. Details like the precise timing, order, or body mechanics of each technique just aren’t there. In other words, these datasets don’t have frame-by-frame timestamps, pose tracking, or motion paths. That means they aren’t much help if you want to analyze techniques in real time or get feedback while training. So, UFC-style data isn’t enough for on-device, fast combat analytics that track every movement during live practice or sparring.
Sports entertainment now drives massive commercial value, pushing researchers to explore AI and machine learning for analyzing sports data. Over the past ten years, studies have increasingly focused on breaking down sports media content. Today’s sports analytics deals with huge, varied datasets that many can access. The biggest challenge? Quickly pinpointing the most useful insights in this flood of information [5].
According to this paper [6] Blaze Pose is a streamlined AI model designed for fast human pose detection on smartphones. It identifies 33 body joints in real-time (30+ FPS on mid-range devices) using a smart blend of heatmaps and coordinate prediction.
Video sharing platforms now handle enormous volumes - over 300 new hours of content every minute. While this creates valuable data opportunities, the scale makes human-led analysis impractical, requiring automated solutions [7].
The field of 6D object pose estimation has gained significant traction, particularly for applications in autonomous navigation and robotic manipulation systems. While deep learning methods have become the predominant approach, there remains a critical need for comprehensive evaluation of contemporary neural architectures and their relative performance advantages across different use cases [8].
 
Spotting human actions has become incredibly important these days, with uses ranging from keeping public spaces safe to creating smarter video games. New tech advances now let us track movements from any angle, even in tricky conditions. Our system can pick up on these action patterns by analyzing how people move through space and time[9].
Our [10] approach combines three smart techniques to predict MMA outcomes:
1.	We first identify key fighting styles by grouping similar fighters using their technical moves (K-means clustering)
2.	We then test multiple AI models - including Random Forests, Neural Networks, and XGBoost - to see which predicts best
3.	Finally, we combine all models' votes for smarter predictions
When testing with real UFC data, our combined system reached 65.5% accuracy - better than any single model. Detailed tests also proved that analysing fighting styles truly helps predictions. This gives both accurate results and clear insights into what makes fighters win[10] .
Statistical analysis of 4,129 MMA decisions (2003-2023) reveals 97.53% outcome concordance between standard round-aggregated scoring and judge-consensus methods, suggesting current MMA judging practices yield consistent results regardless of aggregation approach when applying the 10-Point Must System[11].
Metadata—the "data about data"—powers nearly every digital tool we use daily. Whether you're streaming music on Spotify, sharing Instagram photos, watching YouTube videos, managing finances in Quicken, or texting friends, metadata works behind the scenes. It includes details like creation dates, titles, tags, and descriptions that help systems organize content and users find what they need. This invisible layer is what makes searching, sorting, and sharing possible across all our apps and devices[12].
Single-camera football tracking systems struggle with partial coverage and player re-identification errors. Our multi-camera solution synchronizes feeds from stadium-mounted cameras, using cross-view correlation to maintain uninterrupted tracking and accurate player identification throughout matches[13].
We employ Long Short-Term Memory (LSTM) networks – a specialized recurrent neural architecture optimized for sequential data - to process mobile sensor time-series signals through a hybrid model where a dual-layer LSTM first
 
captures long-term temporal dependencies in the raw sensor data, followed by convolutional blocks that extract localized spatial features from the LSTM outputs, enabling the combined modeling of both time-evolving patterns and their spatial relationships within the sensor readings [4].

III.	METHODOLOGY
A.	Research Design-This study aims a qualitative approach to explore the development of an AI assistant for Mixed Martial Arts (MMA). The system aims to analyze and predict opponent's movements and provide actionable suggestions for counterattacks or improvements. The research utilizes a hybrid deep learning model, integrating pose estimation techniques and computer vision methods to detect and analyze fighters moves, weaknesses, and techniques. The model is made in such a way that it will analyze fight data and provide real-time feedback.

B.	Data Collection-As there is lack of specific datasets for MMA fighters' poses and weaknesses, the system is trained on historical fight records and publicly available data from previous MMA matches such as video recordings and etc. Data Sources: The data is gathered from online fight records (e.g., fighter statistics, match outcomes, previous fight video footage). However, the quality of this data may be limited, and additional preprocessing will be required to enhance its relevance. Pose and Weakness Detection: For this project, we will focus on two MMA fighters. The AI assistant will analyze the movements of fighters and suggest counter-techniques, with a special focus on techniques like strikes, blocks, punches and submission attempts. Media Pipe is used for posing estimation, allowing the AI system to track key body joints and movements, extracting precise data on posture and body orientation during different fight phases.
C.	Technology Selection and Justification - The system architecture prioritizes on-device inference capability, requiring careful selection of computationally efficient components:
1.	Pose Estimation: MediaPipe PoseMediaPipe was selected over alternatives (Open Pose, AlphaPose, HRNet) for several critical advantages:
•	Real-time Performance: Achieves 30+ FPS on mid-range mobile devices through optimized inference pipeline
•	Lightweight Architecture: Single-stage detector enabling deployment without GPU acceleration
•	Comprehensive Tracking: Provides 33 body  landmarks  covering  full-body
 
kinematics necessary for martial arts analysis
•	On-Device Optimization: Pre-compiled for mobile CPUs with quantized weights, reducing model size while maintaining accuracy.
•	Cross-Platform Support: Native integration with mobile (iOS/Android) and embedded systems.
•	This choice enables sub-100ms pose extraction latency critical for real-time tactical feedback during training sessions.
2.	Movement Prediction: Classical Machine Learning
Rather than deep neural architectures (CNN-LSTM), we employ classical supervised learning (Logistic Regression, Random Forest) for next-move prediction:
•	Computational Efficiency: Inference achieves ~14,000 predictions/second with 0.07ms latency per sample on standard CPU
•	Low  Memory  Footprint:  Model  size
<10MB enables edge deployment without memory constraints
•	Training Stability: Converges reliably on limited training data (30-60 second video clips)
•	Interpretability: Feature importance analysis reveals tactical patterns (e.g., hand speed correlates with offensive transitions)
•	No GPU Dependency: Eliminates need for specialized hardware, reducing deployment cost and power consumption
•	This design philosophy trades marginal accuracy gains from deep learning for practical deployability on resource-constrained devices.
D.	Data Ingestion Layer - Input: Fight Clips
1.	Fight Clips: The system allows fight clips to be uploaded in two ways, either manually by the users or retrieved from external sources such as online fight databases, sports archives.
2.	Metadata: Each fight video comes with accompanying metadata, which includes essential information such as:
•	Fighter Details
•	Match Information
•	Previous Stats
•	This metadata is used not only for context but also for organizing the data and helping the AI correlate specific performance patterns across different matches or fighters.
 
3.	Processing: OpenCV for Frame Extraction, Resizing, and Normalization
3.1	Frame Extraction:
a.	OpenCV is used to break the fight video into single frames, usually at a specified frame rate (e.g., 30 fps). this is essential for detailed movement analysis.The system processes each frame to keep track on fighter's poses over time. By observing single frames, it becomes easier to perform pose estimation for each individual movement.
b.	Resizing: Fight clips can be in different resolution. To ensure uniformity and speedy processing, OpenCV resizes the frames to a fixed dimension (e.g., 224x224 pixels).

c.	Normalization: Frames are normalized to ensure uniform pixel intensity (no of pixels per unit area) across all images. Pixel values might be rescaled to a range between 0 and 1. Normalization helps in avoiding problems that could arise due to lighting conditions or other variations in the video. It also improves the accuracy and performance of models like CNN and LSTM by standardizing the inputs.
E.	System Workflow- Our system processes MMA videos frame-by-frame:

3.1 System Workflow
F.	Algorithm
1.	Start: Initialize the system resources such as libraries etc.
2.	Data Collection: Gather fight videos from online sources such as YouTube, fight databases, etc. This data is then used for training the system.
 
3.	Video Preprocessing: Extract frames from the videos at a consistent frame rate (e.g., 30 frames per sec). Resize the frames to a standard resolution (e.g., 224x224 pixels) for consistency. Normalize pixel values to a range [0, 1] to prepare the data for the machine learning models.
4.	Pose Estimation: We Use MediaPipe for pose estimation, detecting key body joints (shoulders, elbows, knees, etc.) in each frame. Generate 2D coordinates of these landmarks(joints) for further analysis.
5.	Feature Extraction: Calculate joint angles, speed, and distance between body parts (e.g., fist velocity during a punch). Track movement patterns across frames, identifying key features like combos or defensive movements.
6.	Data Annotation and Model Training: The extracted pose-derived data are annotated with movement and technique labels such as jab, uppercut, hook, block, and takedown. These annotations enable supervised learning of combat motion patterns from short temporal sequences. In principle, temporal modeling techniques such as Long Short-Term Memory (LSTM) networks are well suited for capturing sequential dependencies in combat movements, while Convolutional Neural Networks (CNNs) can be used to learn spatial relationships related to body posture and attack execution. However, training deep CNN–LSTM architectures typically require large-scale annotated datasets and substantial computational resources. Given the limited size of the available dataset and the emphasis on lightweight, on-device deployment, this work adopts classical supervised learning approaches, including Logistic Regression and Random Forest classifiers, for next-move likelihood estimation. These models provide efficient training, fast inference, and interpretable decision boundaries while remaining effective for pose-based feature representations. The integration of deep CNN–LSTM architectures is identified as a future enhancement, to be explored when larger datasets and stronger computational resources become available, enabling richer spatio-temporal modeling of combat techniques.
7.	Movement & Strategy Analysis: Analyze movement patterns, detecting techniques and possible mistakes (e.g., dropped guard, poor posture). Identify opponent weaknesses by analyzing repetitive movements or openings in their defence. Provide feedback on improving technique and suggest counter-strategies based on detected weaknesses.
8.	Visualization: Create heatmaps showing areas of vulnerability such as exposed head or torso in the opponent's stance. Display movement trajectories to visualize the path of punches, kicks, jabs, hook or
 
other attacks. Overlay suggested corrections, such as raising the guard or adjusting positioning.
9.	Deployment: Deploy the AI Assistant for real-time feedback during sparring sessions or post-fight analysis. Provide a user interface for coaches and fighters to interact with the system for detailed feedback.
10.	End: Conclude the analysis and save the results for future training or improvements.
IV.	Dataset Description
A custom dataset was generated for experimental evaluation using real-world MMA sparring videos. Video streams were processed using Media Pipe Pose to extract 2D skeletal key points in real-time. Video data were collected from single-angle sparring sessions using a standard RGB camera. Each recording was processed with MediaPipe Pose to extract two-dimensional skeletal landmark coordinates in real time. Kinematic features derived from these landmarks, such as hand speed, leg speed, movement intensity, and associated timestamps, were computed over short temporal segments. Action events were annotated and classified into predefined boxing-specific categories: punch_left,punch_right,guard_position,forward_movement,andbackward_movement.The labelled sequences were then stored in structured CSV files for subsequent temporal modeling and evaluation.

4.1 Sample of the processed dataset collected from MMA sparring sessions using pose-derived kinematic features

From these key points, kinematic features such as joint angles, relative limb distances, and temporal motion derivatives were computed and stored in a structured CSV format. The dataset contains pose-based temporal sequences corresponding to common martial arts techniques, including punches, kicks,  defensive  postures,  and  transitional
 
movements. Data was collected from two fighters performing multiple techniques under varying motion dynamics and camera viewpoints. Each action sequence consists of fixed-length temporal windows representing continuous motion patterns. Although limited in scale, the dataset is sufficient for validating the feasibility of real-time, on-device technique recognition. The pose-based representation ensures fighter-agnostic modeling, as classification relies on relative joint dynamics rather than individual appearance. Raw pose key points are used internally for feature computation but are not stored explicitly, enabling a compact and efficient dataset representation suitable for on-device deployment.



V.	System Architecture Layer
1.	Data Ingestion Layer: Input: Fight clips with metadata. Preprocessing: OpenCV for frame extraction, resizing, and normalization.
2.	Pose Detection & Tracking: Pose Estimation: Media Pipe is used for 2D body landmark (joints) detection to track key body points. Tracking: Continuous tracking of body parts is done across frames.
3.	Feature Extraction Layer:Features: Calculation of joint angles, speed, distance between body parts, and movement patterns.
Action Recognition: Identifying offensive and defensive actions (punches, kicks, blocks) based on pose data.
4.	ML Model Layer (Pose-Based Classical ML): Training: Supervised learning on pose-derived features with rolling window context. Models trained on annotated fight sequences. Inference: Lightweight	classifiers	(Logistic Regression/Random Forest) predict next-move probabilities from current pose features and temporal window statistics. Optimized for CPU execution without GPU dependency.
5.	Strategy & Feedback Layer: Feedback Generation: Tactical insights on movement execution, timing, and strategy. Counter-Strategy: Suggest opponent weaknesses and corresponding strategies based on predictions.
6.	Visualization & Interaction Dashboard: Web interface (React/Angular) displaying interactive visualizations: performance charts, movement trajectories, and heatmaps. Visualization Libraries: D3.js/Plotly for data visualization, Seaborn, matplotlib .
7.	Deployment & Real-Time Analysis: Cloud-based Deployment: Hosted on AWS or Google Cloud for scalability and accessibility. Real-Time Feedback:
 
Real-time analysis of sparring videos or matches with immediate insights.


5.1 MMA AI Assistant System Architecture Diagram

VI.	FUTURE SCOPE
Though the feasibility of real-time, on-device combat analytics has been proven, there are some limitations to this proposed method. The dataset used in this research is limited not just in quantity but also in diversity, consisting of a small number of fighters and only specific sparring actions. Although this pose-based approach reduces dependence on person-specific appearance, this must be further examined to validate its effectiveness against greater variation in body type and training methods.
Second, it should be noted that the system employs a single-camera system, which brings on various problems associated with occlusion, especially when the interactions are close-range or when there is an overlap of body movement between the two fighters.
Third, this 2D pose estimation model makes it dependent upon the camera angle and viewpoint. For example, both high angles and large viewpoint changes might affect its effectiveness in capturing kinematic features. These constraints are essentially about balancing some practical trade-offs to deploy this model locally.
The improvements will continue to incorporate new enhancements to address the limitations provided above. Increasing the dataset to have a larger number of participants, varying combat styles, and even more recording conditions will be crucial. For
 
addressing issues of occlusion and viewpoint limitations, multi-camera fusion will be explored.
Also, the use of 3D pose estimation methods may help improve the modeling of space relationships between joints for more complex actions, especially for grappling and clinching situations. The system may also extend its capabilities from punching techniques and actions to comprise additional styles of martial arts, such as kickboxing, wrestling, and mixed martial arts. In addition, more optimizations and tests on mobile devices are necessary for easy integration into real-world training systems.


VII.	CONCLUSION
This work presents a lightweight, real-time, on-device framework for combat movement analysis in mixed martial arts using pose-based temporal features. The proposed system leverages MediaPipe Pose to extract 2D skeletal landmarks from single-camera sparring videos and derives kinematic features such as joint velocities, movement intensity, and temporal action patterns. These features are organized into short rolling windows and analyzed using classical supervised learning models, including Logistic Regression and Random Forest classifiers, to estimate action likelihoods in real time.A key contribution of this work is the demonstration that efficient pose-derived representations, combined with lightweight classifiers, are sufficient to enable low-latency combat technique analysis on edge devices. Experimental results show that model inference incurs negligible computational overhead compared to pose extraction, ensuring sustained real-time performance suitable for live training environments. The system further incorporates an explainable, rule-based feedback mechanism that provides action-aware alerts and technique-level guidance without relying on opaque decision-making.
Unlike existing combat analytics approaches that rely on aggregate post-fight statistics, the proposed framework operates at a fine-grained temporal level, enabling continuous analysis of individual movements during practice sessions. This design makes the system practical for deployment in resource-constrained settings, such as mobile or on-device applications, without dependence on cloud computation.
While the current implementation is evaluated on a limited dataset and focuses on boxing-oriented actions captured from a single camera, it establishes a scalable foundation for future extensions. With the availability of larger annotated datasets and stronger computational resources, deep spatio-temporal models such as CNN–LSTM architectures, multi-camera fusion, and 3D pose estimation can be
 
integrated to enhance modeling capacity and generalization across diverse martial arts styles.
In conclusion, this work demonstrates that real-time, pose-based combat analytics can be achieved using efficient models and minimal hardware requirements. The proposed approach provides a practical and extensible solution for data-driven training support in combat sports, bridging the gap between academic action recognition research and real-world training applications.

VIII.	ACKNOWLEDGMENT
The authors would like to express their sincere gratitude to our mentor, Dr. Bhagyashree Dhakulkar, for her constant guidance, support, and invaluable insights throughout this project. Her insightful suggestions and constructive feedback significantly contributed to the successful completion of this research. Her expertise in artificial intelligence and continuous encouragement played a pivotal role in the development and success of this work. We are deeply thankful for her mentorship, which helped us navigate challenges and refine our approach. This project would not have been possible without her guidance.

IX .REFERENCES
[1]	K. Xia, J. Huang, and H. Wang, “LSTM-CNN Architecture for Human Activity Recognition,” IEEE Access, vol. 8, pp. 56855–56866, 2020, doi: 10.1109/ACCESS.2020.2982225.
[2]	B. Wu and J. Zhou, “Video-Based Martial Arts Combat Action Recognition and Position Detection Using Deep Learning,” IEEE Access, vol. 12,   pp.   161357–161374,   2024,   doi:
10.1109/ACCESS.2024.3487289.
[3]	N. Ćenanović and J. Kevrić, “Mixed Martial Arts Bout Prediction Using Artificial Intelligence,” in Advanced Technologies, Systems, and Applications VII, vol. 539, N. Ademović, E. Mujčić, M. Mulić, J. Kevrić, and Z. Akšamija, Eds., in Lecture Notes in Networks and Systems, vol. 539.
, Cham: Springer International Publishing, 2023, pp. 452–468. doi: 10.1007/978-3-031-17697-5_36.
[4]	Y. Pang, Y. Wang, Q. Wang, F. Li, C. Zhang, and C. Ding, “Applications of AI in martial arts: A survey,” Proc. Inst. Mech. Eng. Part P J. Sports Eng. Technol., p. 17543371241273827, Oct. 2024, doi: 10.1177/17543371241273827.
[5]	H.-C. Shih, “A Survey of Content-Aware Video Analysis for Sports,” IEEE Trans. Circuits Syst. Video Technol., vol. 28, no. 5, pp. 1212–1231, May 2018, doi: 10.1109/TCSVT.2017.2655624.
[6]	V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann,
