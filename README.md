# Relievia ML Backend — Pressure-Injury Risk Prediction Service

A lightweight, high-reliability Python Machine Learning service for **Relievia**, an autonomous smart bed prototype designed for pressure-injury risk monitoring and autonomous repositioning.

---

## 1. Prototype Dataset Notice & Clinical Transparency

> [!IMPORTANT]
> **No Real Clinical Data Found — Prototype Dataset Disclosure**
> The repository was systematically checked for real clinical pressure-injury datasets. As no real clinical dataset exists in this workspace, this prototype uses a **clearly documented synthetic prototype dataset** (`prototype_dataset.csv`).
> 
> This dataset was generated strictly to demonstrate the ML training, evaluation, and serving pipeline. **It must not be construed as verified patient clinical data.**
> 
> The generation parameters are grounded in established clinical tissue perfusion and pressure injury risk factors:
> - **Braden Scale Factors**: Reduced sensory perception/mobility, advanced age ($\ge 65, \ge 75$), and diabetes mellitus (microvascular insufficiency).
> - **Posture-Specific Anatomical Loading**: Supine loading concentrates on sacrum and calcaneus (heels); lateral postures concentrate on the greater trochanter (hip) and acromion (shoulder).
> - **Ischemic Thresholds & Duration**: Unrelieved pressure exceeding capillary closing pressure for $> 90-120$ minutes accelerates tissue hypoxia.

---

## 2. System Architecture & Separation of Concerns

```
+-----------------------------------------------------------------------------------+
|                            Relievia Frontend Simulation                           |
|                               (relievia.html)                                     |
|                                                                                   |
|  1. Collects patient demographics & real-time bed sensor pressure readings        |
|  2. Sends payload to ML Backend: POST /predict                                    |
|                                                                                   |
|                         |  POST /predict              ^                           |
|                         |  (JSON payload)             |  {risk_probability,       |
|                         v                             |   risk_level}             |
|  +----------------------------------------------------+------------------------+  |
|  |               Python Flask ML Service (ml_backend/app.py)                   |  |
|  |                                                                             |  |
|  |  * Normalizes & validates features                                          |  |
|  |  * Runs inference via trained Random Forest model (model.pkl)               |  |
|  |  * ONLY predicts risk — DOES NOT make repositioning decisions              |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                                   |
|  3. Frontend updates Risk Gauge & Digital Twin telemetry                          |
|  4. Relievia JS Decision Logic evaluates if risk threshold (e.g. >= 75%) is met   |
|  5. Relievia JS evaluates candidate postures ('left', 'supine', 'right')          |
|  6. Relievia JS executes motorized 3D bed repositioning & pressure redistribution|
|  7. Sensor values update -> Next simulation cycle queries POST /predict again      |
+-----------------------------------------------------------------------------------+
```

### Strict Architectural Boundaries:
- **ML Backend Role**: Purely analytical risk estimation (`risk_probability` between 0–100, and categorical `risk_level`: `LOW`, `MODERATE`, `HIGH`, `CRITICAL`).
- **Relievia JS Role**: Retains full autonomy over:
  - Determining when intervention is needed.
  - Evaluating candidate postures.
  - Selecting optimal relief angles.
  - Driving the Three.js 3D mechanical bed tilt and patient rotation animations.

---

## 3. Directory Structure

```
ml_backend/
├── app.py                  # Flask REST API with CORS enabled (/predict, /health)
├── predict.py              # Core inference logic: predict_risk(features)
├── train_model.py          # Data generator, Random Forest training & evaluation
├── test_api.py             # Unit tests and HTTP endpoint integration test suite
├── requirements.txt        # Pinned Python package dependencies
├── prototype_dataset.csv   # Documented prototype dataset (generated upon training)
├── model.pkl               # Joblib serialized ML bundle (classifier + regressor + encoders)
└── README.md               # Technical documentation and integration guide
```

---

## 4. Input Features & Output Schema

### Input Features:
| Feature | Type | Valid Values / Examples | Clinical Significance |
|---|---|---|---|
| `age` | Number | `18` - `95` | Dermal thinning, vascular stiffening |
| `mobility` | String | `"immobile"`, `"limited"`, `"independent"` | Spontaneous repositioning capability |
| `diabetes` | Number / Bool | `1` or `0` | Microvascular perfusion impairment |
| `posture` | String | `"supine"`, `"left"`, `"right"` | Anatomical weight-bearing distribution |
| `time_in_current_posture` | Number | Minutes (e.g., `180`) | Cumulative tissue hypoxia duration |
| `sacral_pressure` | Number | `0` - `100` | Primary bony prominence load in supine |
| `heel_pressure` | Number | `0` - `100` | High-risk calcaneal tissue point |
| `hip_pressure` | Number | `0` - `100` | Greater trochanter load in lateral postures |
| `shoulder_pressure` | Number | `0` - `100` | Acromion load in lateral postures |

### Output Response:
```json
{
  "risk_probability": 82.2,
  "risk_level": "CRITICAL"
}
```

Risk levels are mapped according to Relievia clinical thresholds:
- `LOW`: $< 35.0\%$
- `MODERATE`: $35.0\% \le \text{risk} < 60.0\%$
- `HIGH`: $60.0\% \le \text{risk} < 80.0\%$
- `CRITICAL`: $\ge 80.0\%$

---

## 5. Setup & Execution Commands

### Step 1: Install Dependencies
Open PowerShell or your preferred shell and run:
```powershell
python -m pip install -r ml_backend/requirements.txt
```

### Step 2: Train the Random Forest Model
To generate the prototype dataset, train the Random Forest models, and produce `model.pkl`:
```powershell
python ml_backend/train_model.py
```

**Model Performance (20% Holdout Test Set):**
- **Accuracy**: $89.6\%$
- **Precision (Macro)**: $90.1\%$
- **Recall (Macro)**: $89.2\%$
- **F1-Score (Macro)**: $89.4\%$
- **Continuous Score MAE**: $2.74$ points ($R^2 = 0.980$)

### Step 3: Start the Flask API
```powershell
python ml_backend/app.py
```
The server will start on `http://127.0.0.1:5000` (or `http://0.0.0.0:5000`).

---

## 6. How to Test the API

### Option A: Run the Automated Test Suite
With the server running in one terminal (or standalone), run:
```powershell
python ml_backend/test_api.py
```
This runs both direct function unit tests and HTTP REST tests against `/health` and `/predict`.

### Option B: Test via `curl`
**Linux / macOS / Git Bash:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "mobility": "immobile",
    "diabetes": 1,
    "posture": "supine",
    "time_in_current_posture": 180,
    "sacral_pressure": 75,
    "heel_pressure": 60,
    "hip_pressure": 55,
    "shoulder_pressure": 40
  }'
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"age": 72, "mobility": "immobile", "diabetes": 1, "posture": "supine", "time_in_current_posture": 180, "sacral_pressure": 75, "heel_pressure": 60, "hip_pressure": 55, "shoulder_pressure": 40}'
```

**Windows Command Prompt (CMD) / curl.exe:**
```cmd
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"age\": 72, \"mobility\": \"immobile\", \"diabetes\": 1, \"posture\": \"supine\", \"time_in_current_posture\": 180, \"sacral_pressure\": 75, \"heel_pressure\": 60, \"hip_pressure\": 55, \"shoulder_pressure\": 40}"
```

**Expected Response:**
```json
{
  "risk_level": "CRITICAL",
  "risk_probability": 82.2
}
```

---

## 7. Future Integration with `relievia.html`

In future iterations, `relievia.html` can connect directly to `http://127.0.0.1:5000/predict` using standard browser `fetch()`. CORS is already fully enabled on the backend.

### Integration Blueprint for JavaScript:

```javascript
// Function to fetch ML risk score from Python backend
async function fetchMlRiskPrediction() {
  const sc = state.sc; // Patient scenario
  
  // Map simulated bed pressure zones to standard model features
  const payload = {
    age: sc.age,
    mobility: sc.mobility.toLowerCase(), // 'bedbound' -> mapped to 'immobile'
    diabetes: sc.diabetic ? 1 : 0,
    posture: state.posture,             // 'supine', 'left', or 'right'
    time_in_current_posture: state.timeInPosture,
    sacral_pressure: state.pressure.lowerback || state.pressure.upperback || 50,
    heel_pressure: state.pressure.heels || 40,
    hip_pressure: Math.max(state.pressure.lhip || 0, state.pressure.rhip || 0),
    shoulder_pressure: Math.max(state.pressure.lshoulder || 0, state.pressure.rshoulder || 0)
  };

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const data = await response.json();
      // Apply the ML risk prediction to the live UI state
      state.riskScore = Math.round(data.risk_probability);
      renderGauge(state.riskScore);
      renderTwinFacts(state.riskScore);
      
      // Existing Relievia JS decision logic remains in full control:
      if (state.riskScore >= RISK_THRESHOLD && !state.criticalLogged) {
        state.criticalLogged = true;
        logTimeline('ML Risk elevated to ' + data.risk_level + ' (' + state.riskScore + '%)', 'crit');
        beginEvaluation(); // JS autonomously initiates bed repositioning
      }
    }
  } catch (err) {
    console.warn("ML backend unreachable, continuing on local simulation logic:", err);
  }
}
```
