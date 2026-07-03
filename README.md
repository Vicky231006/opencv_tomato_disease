# AgTech Vision OS: Distributed Edge AI Crop Disease Diagnostic & Severity Inference Engine

An enterprise-grade, full-stack Edge AI and computer vision system engineered for low-latency detection, multi-class classification, and mathematical surface-area severity quantification of crop pathologies. 
The system implements a decoupled, multi-stage deep learning pipeline leveraging a **MobileNetV3** gatekeeper network, a fine-tuned **ResNet50** classification core optimized with discriminative learning rates, and a custom **multi-channel HSV color-space geometric segmentation engine**. The system serves real-time analytics over an asynchronous **FastAPI** REST layer to a responsive **React / Tailwind CSS v4** single-page management dashboard.
---
## 🚀 Live Demonstration

Below is a live walkthrough of the diagnostic panel in action, showcasing the image ingestion, scanning laser animations, real-time pipeline telemetry, and disease care recommendations:

<br />

<div align="center">
  <video src="demo_cv.mp4" width="100%" controls loop muted playsinline></video>
</div>

<br />
---
## 🏗️ System Architecture & Inference Pipeline Topology

Instead of passing raw user uploads directly into heavy deep learning models, the architecture utilizes a **Decoupled Sequential Pipeline Pattern**. This structural decision optimizes system memory, drops unnecessary GPU tensor processing costs, and implements strict verification boundaries before running deep inference.
```mardown
                [ Raw Specimen Asset Input (JPG/PNG) ]
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  Stage 1: Deterministic QC   │ ──(Fails Blur/Lighting)──► [ 422 Unprocessable ]
                 └───────────────────────────────┘
                                 │ (Passed)
                                 ▼
                 ┌───────────────────────────────┐
                 │ Stage 2: MobileNetV3 Verifier │ ──(Not a Leaf Structure)──► [ 422 Unprocessable ]
                 └───────────────────────────────┘
                                 │ (Leaf Confirmed)
                                 ▼
                 ┌───────────────────────────────┐
                 │ Stage 3: ResNet50 Classifier  │ ──(Extracts Disease Label)
                 └───────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   [ Pathogen Detected ]                   [ Foliage Healthy ]
              │                                     │
              ▼                                     ▼

┌─────────────────────────────────┐            ┌────────────────┐
│ Stage 4: Multi-Channel HSV      │            │ Bypasses Mask  │
│ Geometric Area Segmentation     │            │ Severity = 0%  │
└─────────────────────────────────┘            └────────────────┘
                │                                     │
                └──────────────────┬──────────────────┘
                                   ▼
                    [ Structured JSON REST Response ]
                    [ (Class, Severity %, Protocols) ]

```
### 1. Stage 1: Deterministic Quality Control Layer (OpenCV)
The incoming image payload buffer is ingested into an OpenCV matrix. The engine converts the matrix to a single grayscale channel to compute light distribution and focus metrics:
* **Focus Metric:** Computed via the variance of the Laplacian operator ($\sigma^2$). Images displaying $\sigma^2 < 30.0$ indicate excessive motion blur or macro-focus failures and are rejected.
* **Luminance Metric:** Evaluated using global mean pixel intensity ($\mu$). Payloads with $\mu < 15$ (severe underexposure/shadowing) or $\mu > 245$ (sensor clipping/overexposure) are rejected immediately to protect downstream model reliability.

### 2. Stage 2: Binary Object Gatekeeper (MobileNetV3 Small)
* **Objective:** Prevents out-of-domain payloads (e.g., random objects, background noise, or incorrect crop components) from flooding the main disease classifier.
* **Topology:** A lightweight `MobileNetV3-Small` architecture initialized with ImageNet parameters, modified with a custom linear layer head targeting 2 output nodes (`leaf` vs. `non_leaf`).
* **Trade-off & Execution Performance:** Chosen specifically for its inverted residual blocks and squeeze-and-excitation modules, achieving near-perfect validation accuracy while keeping parameter counts low enough to prevent CPU pipeline bottlenecks.

### 3. Stage 3: Fine-Grained Feature Extraction & Pathology Classification (ResNet50)
* **Objective:** Maps the leaf specimen to its specific pathological taxonomy out of 10 distinct target target profiles.
* **Topology:** A `ResNet50` architecture. While early model attempts frequently confused highly visual edge cases like **Tomato Early Blight** (Alternaria solani) and **Tomato Late Blight** (Phytophthora infestans) due to overlapping global color profiles, the production model unfreezes the deep residual blocks (`layer3` and `layer4`) to force the network to isolate micro-textural features.

### 4. Stage 4: Multi-Channel HSV Geometric Segmentation Engine
* **Architectural Trade-Off Decision (Ditching Deep Segmentation Models):** Instead of using dense, resource-heavy instance segmentation networks (like Mask R-CNN or YOLOv8-seg) which require tens of thousands of manually clicked polygon annotations and slow down GPU processing times, this pipeline relies on an **optimized, rule-based mathematical computer vision matrix pattern**. 
* **Methodology:** The system runs a concurrent multi-channel HSV thresholding process to isolate expanding chlorotic halos and deep necrotic structures simultaneously. It overlays these directly onto the isolated leaf mask boundaries, instantly calculating the pixel-perfect surface area percentage ratio of the disease:
$$\text{Severity Percentage} = \left( \frac{\text{CountNonZero}(\text{Spot Mask} \cap \text{Leaf Structure})}{\text{CountNonZero}(\text{Leaf Structure})} \right) \times 100$$
* This approach calculates precision metrics instantly on low-power host CPUs without needing any extra deep training steps.

---

## 📊 Deep Learning Model Training Analytics

### Model 1: Leaf Gatekeeper (MobileNetV3)
* **Optimization Strategy:** Standard Stochastic Gradient Descent with an initial learning rate ($\eta = 0.001$).
* **Loss Function:** Cross-Entropy Loss.
* **Convergence Evaluation:** The network achieved stable optimization within 3 epochs, hitting perfect validation performance.

```text
Starting Training Execution Loop for leaf_verifier.pth...
Epoch [1/3] | Train Loss: 0.0189 | Train Acc: 98.91% | Val Acc: 99.58%
Epoch [2/3] | Train Loss: 0.0060 | Train Acc: 99.95% | Val Acc: 99.79%
Epoch [3/3] | Train Loss: 0.0008 | Train Acc: 99.95% | Val Acc: 100.00%
Saved configuration weights as leaf_verifier.pth
```
### Model 2: Advanced Disease Classifier (ResNet50)
* **Optimization Strategy:** Adam Optimizer utilizing **Discriminative Learning Rates** to update model weights without washing out pre-trained ImageNet spatial features.
* **Learning Rate Layer Allocation Strategy:**
    $$\eta_{\text{linear\_head}} = 10^{-3}, \quad \eta_{\text{layer4}} = 10^{-4}, \quad \eta_{\text{layer3}} = 10^{-5}$$
* **Loss Function:** Cross-Entropy Loss.
* **Convergence Evaluation:** The network reached an impressive **97.85% validation accuracy** by Epoch 5, proving that unfreezing the lower feature layers successfully taught the model to tell the difference between Early Blight concentric rings and Late Blight water-soaked lesions.

```text
Initializing Advanced Tomato Disease Classifier (ResNet50)...
Starting Training Execution Loop for tomato_classifier.pth...
Epoch [1/5] | Train Loss: 0.2572 | Train Acc: 91.74% | Val Acc: 98.41%
Epoch [2/5] | Train Loss: 0.0436 | Train Acc: 98.50% | Val Acc: 99.19%
Epoch [3/5] | Train Loss: 0.0396 | Train Acc: 98.64% | Val Acc: 97.50%
Epoch [4/5] | Train Loss: 0.0260 | Train Acc: 99.19% | Val Acc: 98.72%
Epoch [5/5] | Train Loss: 0.0363 | Train Acc: 98.83% | Val Acc: 97.85%
Saved configuration weights as tomato_classifier.pth
```
---

## 💻 Tech Stack & Production Implementation Matrix

### Backend Core (Inference Layer)
* **Framework:** FastAPI (Asynchronous Python ASGI Server)
* **Tensor Engine:** PyTorch (Torchvision Model Zoo)
* **Matrix Processing:** OpenCV (cv2) & NumPy
* **Deployment Worker Host:** Uvicorn

### Frontend Suite (Analytical Dashboard)
* **Core Engine:** React.js (Single-Page UI Engine)
* **Style Compiler:** Tailwind CSS v4 (Native Vite Plugin Integration)
* **Build Pipeline Utility:** Vite

---

## 🛠️ Installation & Local System Deployment

### Prerequisites
* Python 3.10+
* Node.js (v18+)
* NVIDIA CUDA Driver Toolkit (Optional, defaults to CPU execution cleanly if missing)

### 1. Backend Server Setup
To execute the server setup, configure your virtual environment and run the package installer engine:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
# Clean package ingestion run using the requirements manifest
```python
pip install -r requirements.txt
```
Make sure your trained weights binaries are copied directly into the backend directory:

backend/
├── app.py
├── pipeline_module.py
├── requirements.txt
├── leaf_verifier.pth
└── tomato_classifier.pth

Fire up the asynchronous Uvicorn server worker process:
```python
python app.py
```
The server will boot up and establish its live REST API port listener at http://localhost:8000.

### 2. Frontend Interface Setup
Open a separate terminal window tab, initialize your Vite template wrapper project bundle, and pull down the Tailwind CSS v4 design plugins:

```bash
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite
```

Verify your layout build pipeline by checking that your vite.config.js loads the Tailwind compiler plugin:

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})

Inject the Tailwind import directive into the top of your global src/index.css file:

@import "tailwindcss";

Launch the frontend client local web dev server:
```bash
npm run dev
```
Open your browser window to http://localhost:5173 to access the live dashboard environment.

---

## 📡 Production API Reference Spec

### 1. System Health Check
* **Endpoint:** GET /api/v1/health
* **Description:** Returns the active server running state and indicates whether computation is bound to CPU or CUDA hardware topologies.
* **Response Payload:**

{
  "status": "healthy",
  "device_target": "cuda"
}

### 2. Run Specimen Diagnostics Pipeline
* **Endpoint:** POST /api/v1/diagnose
* **Payload Type:** multipart/form-data
* **Parameter:** file: UploadFile (Supports raw image binaries matching standard JPG, JPEG, or PNG extension configurations)
* **Success Response (200 OK):**

{
  "success": true,
  "data": {
    "crop": "Tomato",
    "condition": "PATHOGEN DETECTED",
    "diagnosed_class": "Tomato Early blight",
    "severity_percentage": 28.45,
    "treatment_protocol": "Use chlorothalonil or copper fungicides. Trim lower branches."
  }
}

* **Rejection Response (422 Unprocessable Entity):**

{
  "detail": "Object validation failed. Target is not a crop leaf structure."
}

---

## 💡 Engineering Insights & Core System Safeguards
* **Smart Memory Caching:** The heavy neural networks load into memory exactly once when the FastAPI server initializes. Submitting new images runs inference immediately without causing performance spikes.
* **Path Inversion Security:** All file path references match the real directory contexts dynamically based on the local system script root location (__file__), preventing path resolution bugs across different OS platforms (like Arch Linux vs. Ubuntu cloud nodes).
* **Context-Aware Severity Masking:** The OpenCV multi-channel severity engine only evaluates pixels that sit within the leaf boundary mask. This ensures that dark floor shadows or light backgrounds are completely ignored, keeping the severity calculation highly accurate.

