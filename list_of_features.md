# Features and Requirements: Particle Filter-Based Multi-Agent System (PF-MAS)

## 1. Project Overview
A Multi-Agent System (MAS) designed to monitor tool degradation during micro-milling of 20 cm steel channels[cite: 1]. The system uses a **Particle Filter (PF)** to predict Remaining Useful Life (RUL) and a **Confidence Interval (CI)** trigger to decide when ground-truth imaging is required[cite: 1].

## 2. Data Structure & Streaming
*   **Channel Segmentation:** The force data from `hardstavaxtool15datalong.txt` is divided into **45 distinct channels** representing the 20 cm lines[cite: 1].
*   **Downsampling:** High-frequency data is downsampled at a 1:20 ratio to maintain dashboard performance[cite: 1].
*   **Iterative Arrival:** Data arrives channel-by-channel, simulating a real-time production environment[cite: 1].

## 3. Agent Architecture

### 3.1 Leader Agent (Decision & Trigger)
*   **Anomaly Trigger:** When a new force data point arrives and falls **outside the previous Prediction Confidence Interval**, the Leader Agent triggers the Image Agent[cite: 1].
*   **Decisions:**
    *   **Continue:** If the data point is within the CI[cite: 1].
    *   **Take Image:** If the data point is outside the CI (Model Mismatch)[cite: 1].
    *   **Replace:** If the PF predicts a >90% probability of "Critical Status"[cite: 1].

### 3.2 Force Agent (Probabilistic Prediction)
*   **Algorithm Integration:** Utilizes `ForceDataAnalysisv11tool15.m`, `ForceDataAnalysisv13tool15.m`, and `run_Spline_Optimizer.m`[cite: 1].
*   **Prediction Engine:** Uses a Particle Filter to predict the tool wear trajectory until a critical threshold is reached[cite: 1].
*   **Uncertainty Modeling:** Dynamically calculates the **Confidence Interval** for the next expected data point based on current particle weights[cite: 1].

### 3.3 Image Agent (Ground Truth Calibration)
*   **Feature Extraction:** Utilizes `wornarea.m` to calculate tool radius and actual worn area[cite: 1].
*   **Geometric Validation:** Implements the Ideal Worn calculation:
    $$A_{ideal} = R_{fit}^2 \left( \cot\left(\frac{\theta}{2}\right) - \frac{\pi - \theta}{2} \right)$$
*   **K-Means Classification:** Uses a 3-centroid K-Means model (trained on `Tool_Features_Dataset.xlsx`) to categorize the image into **Factory New**, **Mid-Worn**, or **Critical Status**[cite: 1].
*   **PF Reset:** Feeds this ground truth back to the Force Agent to "collapse" the particles and recalibrate the prediction[cite: 1].

## 4. Dashboard Requirements (Python-Native)
*   **Real-Time Plotting:** Displays iterative force arrivals and the evolving Particle Filter "cloud"[cite: 1].
*   **Confidence Visuals:** Shows the prediction trajectory and the shaded Confidence Interval[cite: 1].
*   **Image Feed:** Displays processed images from the `TestData` folder (mapped to the current channel) when an image decision is made[cite: 1].
*   **State Monitor:** Shows the K-Means cluster result and current RUL prediction[cite: 1].