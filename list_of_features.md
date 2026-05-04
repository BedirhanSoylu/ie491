# Features and Requirements Specification: Multi-Agent Tool Wear Monitoring System

## 1. Project Overview
The objective is to build a Multi-Agent System (MAS) to monitor and predict the degradation of a micro-milling cutting tool used on hardened steel[cite: 1]. The system must optimize tool utilization by deciding whether to continue production, pause for an image-based inspection, or replace the tool based on high-frequency force data and intermittent image analysis[cite: 1].

## 2. Agent Architecture & Operations

### 2.1 Leader Agent (The Decision Maker)
*   **Role:** Acts as the central coordinator that processes inputs from all sub-agents to make a dynamic decision: **Continue**, **Stop for Image**, or **Replace Tool**[cite: 1].
*   **Logic:** Employs an advanced, non-rule-based approach (e.g., Reinforcement Learning or a Markov Decision Process) to balance the trade-off between the high cost of stopping/imaging and the risk of tool failure[cite: 1].
*   **Dynamic Weighting:** Must weigh the "noisy" high-frequency force trends against the "ground truth" but high-cost image data[cite: 1].

### 2.2 Force Data Agents (Real-Time Sub-Agents)
These agents process streaming force data in the X and Y directions[cite: 1].
*   **Amplitude & Zero-Point Analyst (`ForceDataAnalysisv11tool15.m`):** Analyzes the amplitude of minimum force and identifies 0-force points to detect cutting stability[cite: 1].
*   **Histogram Analyst (`ForceDataAnalysisv13tool15.m`):** Provides a statistical distribution (histogram) of force data to identify shifts in cutting behavior[cite: 1].
*   **Normal/Tangential Force Analyst (`run_Spline_Optimizer.m`):** Converts x-y coordinate force data into normal and tangential magnitudes to detect the rapid "plateau and spike" behavior indicative of end-of-life[cite: 1].

### 2.3 Image Analysis Agents (Intermittent Sub-Agents)
These agents process high-resolution images of the cutting tool after specific channels are completed[cite: 1].
*   **Tool Geometry Analyst (`wornarea.m`):** Calculates tool radius, corner radius, and the specific worn tool area[cite: 1].
*   **Ideal Wear Calculator:** Implements the geometric equation to calculate the theoretical ideal worn area:
    $$A_{ideal} = R_{fit}^2 \left( \cot\left(\frac{\theta}{2}\right) - \frac{\pi - \theta}{2} \right)$$
*   **Wear Gap Detection:** Compares the actual worn area (from `wornarea.m`) against the geometric ideal wear to determine the tool wear increment[cite: 1].

## 3. Data Pipeline & Processing
*   **Streaming Simulation:** The system must simulate real-time data by reading `hardstavaxtool15datalong.txt`[cite: 1].
*   **Downsampling:** To maintain system performance during high-frequency streaming, the data must be downsampled by taking 1 point for every 20-point interval[cite: 1].
*   **Feature Categorization:** Predicted tool life must be clustered into three primary statuses: **Factory New**, **Mid-Worn**, and **Critical Status**[cite: 1].
*   **Training Data:** Use `Tool_Features_Dataset.xlsx` (unlabeled feature extraction data) and images in the `TestData` folder for clustering and model training[cite: 1].

## 4. Dashboard & Visualization
*   **Framework:** Built using a Python-native framework (Streamlit or Plotly Dash) capable of high-speed updates[cite: 1].
*   **Live Metrics:** Real-time visualization of streaming force data and the calculated normal/tangential magnitudes.
*   **Status Indicators:** Visual indicators of the current tool status (New/Mid/Critical) and the Leader Agent's current decision.
*   **Image Gallery:** Display of the latest tool images from the `TestData` folder when an "Image" decision is triggered.