# PROJECT AEGIS: Full Technical Context & System Report
### Adversarial Immune System for GenAI-Era Payment Fraud Prevention
**Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026**

---

## 1. Executive Summary

**Project AEGIS** is a closed-loop, adversarial red-team/blue-team AI platform engineered to secure real-time payment architectures against generative AI-driven fraud. AEGIS systematically addresses fraud across four major attack surfaces: **unstructured text (NLP/AML), transaction graphs (identity relations), behavioral biometrics (telemetry bypass), and agentic commerce (autonomous execution hijacking)**. 

To demonstrate production feasibility under real-world low-latency requirements, AEGIS implements a **two-speed architecture**:
1. **Synchronous Edge (<15ms target):** A compiled C++ transaction simulator streams live transactions over high-speed sockets (TCP/ZeroMQ) into a lightweight Python FastAPI service that executes a calibrated XGBoost tabular classifier.
2. **Asynchronous Analytical Core:** Simultaneously forks categorical, text, and graph payloads to deep analytical pipelines (including NetworkX-based centrality graph profiling, sentence-transformer semantic embedding alignment, and Kolmogorov-Smirnov biometric variance testing) to update merchant risk topologies and terminal profiles post-transaction.

By isolating the low-latency authorization decision from the heavier structural and semantic calculations, AEGIS achieves sub-50ms live round-trip latency on mixed legitimate and adversarial transaction streams while maintaining high detection precision.

---

## 2. Challenge Context & Evaluation Criteria

The **Mastercard Innovation Challenge — AI Defense Lab for Payment Security** is hosted at the **Global Fintech Fest (GFF) 2026** (September 9–11, Jio World Centre, Mumbai). The challenge tasks participants with building an end-to-end AI system structured across three core pillars:
*   **Identify (Ideate):** Exhaustively map emerging, novel GenAI-powered payment fraud vectors.
*   **Generate (Simulate):** Build algorithms and agents to simulate those attacks at scale with statistical fidelity.
*   **Defend (Mitigate):** Build a production-grade machine learning and statistical classifier to detect, flag, and mitigate those attacks in real time.

Submissions are evaluated against five official dimensions, each mapped directly to an AEGIS system mechanism:

| Evaluation Criterion | What Judges Are Testing | AEGIS System Mechanism |
| :--- | :--- | :--- |
| **Diversity of Attacks** | Breadth across rails, channels, and surfaces (not minor variations of one concept) | **9 distinct vectors** spanning Reconnaissance/Infra, Weaponization, and C2/Exfiltration stages of the attack kill chain. |
| **Fidelity of Simulation** | Realistic statistical properties of fuzzed/synthetic datasets | **Distribution-anchored sampling** from the Kaggle IEEE-CIS base dataset, Gaussian Mixture Model biometric noise injection, and diurnal temporal jitter. |
| **Detection Efficacy** | Rigorous metrics on unseen datasets rather than cherry-picked demo success | Class imbalance correction via **SMOTE**, Platt/Isotonic **score calibration**, and evaluation on a strict **held-out-by-entity validation split**. |
| **Novelty** | Genuinely new systems/algorithmic ideas rather than basic out-of-the-box wrappers | **C++ decoupled routing architecture** combined with a **closed-loop adversarial feedback loop** (using GAN-as-perturbation-search). |
| **Real-world Feasibility** | Real-time performance, false decline mitigation, and compliance alignment | **Live-measured latency speedometers** in Streamlit, **three-zone friction controls** (Allow/Step-Up/Block), and integration of a **human-review queue** and RBI regulatory guidelines. |

---

## 3. Dual-Track Strategy

To maximize scoring and survive technical cross-examination, AEGIS employs a structured **Dual-Track Strategy** that separates the visionary future architecture from the buildable, shippable proof-of-concept (PoC).

### Track A: The Enterprise Vision (The Pitch Deck)
Track A articulates a comprehensive enterprise integration designed around Mastercard’s core fraud-prevention platforms:
*   **Decision Intelligence (DI) Pro:** Targeted by Graph Poisoning and Sleeper Mule attacks. Employs advanced real-time relational risk scoring to connect retail nodes and isolate networks rather than relying on static rules.
*   **NuData:** Targeted by Biometric Latent Diffusion Mimicry. Leverages passive, multi-horizon behavioral biometrics (keystroke cadence, screen pressure, device angle) to authenticate users invisibly.
*   **Agent Pay / Web Bot Auth:** Targeted by Agentic Token Hijacking and Prompt Injection. Secures emerging agentic commerce interfaces and delegated token authentication flows.

### Track B: The Hackathon Execution (The 7-Day Shipped Build)
Track B represents the actual functional codebase implemented during the 7-day hackathon sprint. Every high-level enterprise model in Track A is mapped to a fast, mathematically equivalent, dependency-light approximation in Track B. This is referred to as the **"Enterprise Term" → *(Practical Build)*** convention:

*   ***Masked Graph Autoencoder (GAE)*** → *(NetworkX Centrality Features + Isolation Forest)*
*   ***Temporal Fusion Transformer (TFT)*** → *(Rolling Coefficient of Variation + Kolmogorov-Smirnov Test)*
*   ***Fine-Tuned BERT/RoBERTa Classifier*** → *(TF-IDF + Pretrained Sentence-Transformers Cosine Similarity)*
*   ***Generative Adversarial Network (GAN)*** → *(Parametric Adversarial Perturbation Search)*

By explicitly documenting this split, the team projects technical honesty and rigor, earning trust from technical judges who probe the actual repository code.

---

## 4. Full Attack Taxonomy (The Kill-Chain Model)

Rather than presenting attack scenarios as a flat list, AEGIS maps vectors across a structured **Cyber Kill Chain**, showing how threat actors establish infrastructure, deliver payloads, and exfiltrate funds.

```
+------------------------------------+      +-----------------------------------+      +---------------------------------+
| 1. RECONNAISSANCE & INFRASTRUCTURE | ---> |    2. WEAPONIZATION & DELIVERY    | ---> |   3. COMMAND & CONTROL / EXFIL  |
| - Synthetic Merchant Onboarding    |      | - Generative Graph Poisoning      |      | - Micro-Structuring / Smurfing  |
| - Agentic Token Hijacking          |      | - Biometric Latent Diffusion      |      | - Device Farm / Emulator Clustering|
| - Permission Scope Creep (UPI)     |      | - Agentic Semantic Smuggling      |      |                                 |
+------------------------------------+      +-----------------------------------+      +---------------------------------+
```

### 4.1 Reconnaissance & Infrastructure Attacks
*   **A. Synthetic Merchant Onboarding (Fake Site / Shell Merchant)**
    *   *Mechanism:* Attacker deploys cloned storefronts using a static site generator ("Generative Web Cloning") and registers them with a payment aggregator using synthetic face imagery and fake business documents ("Synthetic Identity Fabrication").
    *   *Target:* Aggregator-level trust bootstrapping, weak automated KYC, and MCC misclassification.
    *   *Defense Concept:* **Merchant Risk Graph Embedding** (*practically: NetworkX similarity clustering on domain age, SSL issuer, hosting IP block, and DOM/CSS hash cosine similarity*). It runs typosquatting checks (Levenshtein distance) of domain names against the top 10,000 retail brands.
*   **B. Agentic Token Hijacking (Agent-in-the-Middle)**
    *   *Mechanism:* A user authorizes an AI shopping agent using a delegated token. An attacker uses **Indirect Prompt Injection** (hidden instructions in HTML/alt-tags on an untrusted page the agent visits) to rewrite the agent’s tool call, redirecting payments.
    *   *Defense Concept:* **Intent-to-Action Divergence Verification** (*practically: Cosine similarity of sentence-transformer embeddings* between user natural-language prompt and final payment API payload fields). Backed by **Action Provenance Chaining** (a Merkle-style hash log of all tools executed; any untrusted domain call flags the chain).
*   **C. Permission Scope Creep / Consent Escalation**
    *   *Mechanism:* A legitimate recurring-payment mandate (such as UPI Autopay or e-NACH) is authorized for small values, but the merchant API silently escalates the frequency/amount without re-consent (Consent Boundary Violation).
    *   *Defense Concept:* **Permission Delta Monitoring** (*practically: Schema diffing* of mandate modification calls; any field change outside a strict allowlist triggers dynamic re-authentication). Cites the **RBI e-mandate framework** (requiring AFA re-auth on modifications or transactions above ₹1 Lakh), signaling deep regulatory literacy.

### 4.2 Weaponization & Delivery Attacks
*   **D. Generative Graph Poisoning ("The Sleeper Mule" - Core Build)**
    *   *Mechanism:* Threat actors create 50 synthetic cards (Tokenized PANs) and route low-value transactions (₹100–300) to a single terminal over a 90-day window to build trust edges, followed by a coordinated bust-out.
    *   *Fidelity Technique:* **Adversarial Graph Perturbation** (*practically: Parameterized NetworkX graph generation with custom edge weights*), forcing the model to generalize across varied topologies.
    *   *Defense Concept:* **Masked GAE** (*practically: Node-level Degree Centrality, Betweenness Centrality, Clustering Coefficient, and PageRank fed into an Isolation Forest*).
*   **E. Biometric Latent Diffusion Mimicry (Core Build)**
    *   *Mechanism:* Adversary synthesizes continuous behavioral telemetry (touchscreen pressure, keystroke speed, phone angle) to bypass passive biometric layers.
    *   *Fidelity Technique:* **Parametric Resampling** (*practically: Fitting a Gaussian Mixture Model (GMM)* to a legitimate user's profile and sampling a spoof). The sampled spoof is clipped of human micro-tremors, making it too mathematically "perfect" and smooth.
    *   *Defense Concept:* **Temporal Fusion Transformer** (*practically: Rolling Coefficient of Variation (σ/μ) paired with a Kolmogorov-Smirnov (KS) test* comparing session variance against historical baselines; yields an interpretable p-value for reason logs).
*   **F. Agentic Semantic Smuggling (Core Build)**
    *   *Mechanism:* Threat actors route transactions under high-risk merchant codes (e.g., MCC 6051 - Crypto) but write innocent descriptions ("Corporate Software Subscription") to slip past lexical AML filters.
    *   *Fidelity Technique:* **Adversarial Text Rewriting** (*practically: local spaCy Named Entity Recognition (NER) + synonym substitution* generating 20-30 B2B invoice variations).
    *   *Defense Concept:* **Fine-Tuned BERT** (*practically: Pretrained Sentence-Transformer `all-MiniLM-L6-v2` embeddings + Cosine Similarity* against an MCC Description Anchor Set; flags when text aligns with a different high-risk MCC description).
*   **G. Synthetic Identity Bust-Out (Flagship Combo Scenario - Core Build)**
    *   *Mechanism:* Chaining Vector A (Fake Merchant Onboarding) with Vector D (Sleeper Mule farming) and Vector F (Semantic Smuggling).
    *   *Significance:* This is the central live demo sequence. A single payment series triggers three models simultaneously, proving the system handles complex, real-world blended attacks.

### 4.3 Command-and-Control & Exfiltration
*   **H. Micro-Structuring / Smurfing (AML Evasion)**
    *   *Mechanism:* Splitting large sums into hundreds of micro-transfers across cascading accounts to stay under regulatory triggers, before consolidating at exit nodes.
    *   *Defense Concept:* **Structuring Detection via Transaction Velocity Graphs** (*practically: NetworkX MultiDiGraph sliding-window network geometry* to detect Fan-Out/Fan-In patterns within 72 hours, independent of amount randomization).
*   **I. Device Farm / Emulated Device Clustering**
    *   *Mechanism:* Emulator farms rotating IMEIs and rotating device fingerprints to bypass unique device binding rules.
    *   *Defense Concept:* **Device Fingerprint Entropy Analysis** (*practically: DBSCAN clustering on device metadata* such as screen resolution, timezone, font hashes; emulator farms group into unnaturally tight clusters compared to high-entropy real-world user devices).

---

## 5. Machine Learning & Statistical Architecture

The backend implements four specialized models, unified by a calibrated risk aggregator:

```
                  +-------------------------+
                  |  C++ SIMULATOR STREAM   |
                  +-------------------------+
                               |
            +------------------+------------------+
            | (Sync Edge)                         | (Async Core)
            v                                     v
+-----------------------+               +-----------------------------------+
|  Tabular XGBoost      |               | - Graph: Isolation Forest (NetX)  |
|  (SMOTE + calibrated)  |               | - Biometric: GMM / KS-test        |
+-----------------------+               | - Text: Sentence-Transformer      |
            |                           +-----------------------------------+
            v                                             |
     xgb_score (0.4)                                      | update context
            |                                             v
            +------------------> [ RISK AGGREGATOR ] <----+
                                         |
                                         v
                         +-------------------------------+
                         | total_risk_score              |
                         | - Allow (< 0.60)              |
                         | - Step-Up (0.60 - 0.85)       |
                         | - Hard Block (> 0.85)         |
                         +-------------------------------+
```

### 5.1 Calibrated Risk Aggregator
Raw ML model outputs are not probability scores by default. AEGIS applies **Platt Scaling / Isotonic Calibration** (via Scikit-learn's `CalibratedClassifierCV`) to ensure the scores behave as true probabilities before combining them into a unified decision matrix:

$$\text{total\_risk\_score} = (xgb\_score \times 0.4) + (graph\_isolation\_score \times 0.3) + (nlp\_cosine\_divergence \times 0.3)$$

### 5.2 Three-Zone Friction Model
*   **Allow (score $\le$ 0.60):** Seamless frictionless checkout.
*   **Step-Up Authentication (0.60 < score $\le$ 0.85):** Triggers dynamic friction (FaceID, SMS OTP). Mitigates false declines by routing ambiguous transactions (e.g., child using a parent’s phone, or weirdly worded B2B invoice) to confirmation layers.
*   **Hard Block (score > 0.85):** Transaction aborted. Logs full SHAP feature attribution and pushes the event to the **Human-in-the-Loop Review Queue** to resolve merchant compliance/liability questions.

### 5.3 Stacking Meta-Learner (Advanced Strategy)
To avoid manual weighting bias (the arbitrary 0.4/0.3/0.3 split), the system builds an alternative **Stacked Generalization Meta-Learner** (Logistic Regression or shallow XGBoost). This model is trained on out-of-fold predictions of the base models to mathematically optimize weighting directly from dataset performance.

---

## 6. System & Pipeline Engineering

AEGIS's architecture decoupled systems to guarantee live throughput limits.

### 6.1 Synchronous vs. Asynchronous Decoupling
To execute GNN and Transformer evaluation within the strict real-time payment authorization window (<50ms):
*   The high-performance **C++ routing engine (`simulator.cpp`)** reads stream entries and instantly queries the **XGBoost tabular model synchronously**.
*   The C++ engine forks the text and graph arrays asynchronously to local queues. The **Graph Centrality Isolation Forest** and **Sentence-Transformer text embeds** run in background threads to update central state, ensuring the current transaction's authorization is never blocked by GNN/NLP latency.

### 6.2 Data Generation & Fidelity Engineering
Fidelity engineering prevents technical judges from spotting simulated, static patterns:
*   **Amounts:** Log-normal spend distributions sampled directly from the Kaggle IEEE-CIS dataset.
*   **Timestamps:** Realistic daypart-weighted distributions (coffee shops transaction peaks cluster 7–10am and 1–3pm).
*   **Graph Structure:** Legitimately sparse-but-branching merchant ego-graphs are blended alongside the sleeper mule nodes, preventing trivial "anomaly isolation".

---

## 7. Operational & Project Timeline

Project AEGIS is designed for rapid execution across a strict 7-day development cycle:

```
Day 1: Foundation      Day 2: Fidelity        Day 3: ML Core         Day 4: Loop & Eval     Day 5: C++ Edge        Day 6: Streamlit UI    Day 7: Submission
+------------------+   +------------------+   +------------------+   +------------------+   +------------------+   +------------------+   +------------------+
| - Downl. dataset |   | - GMM Biometrics |   | - Train XGBoost  |   | - Aggregator &   |   | - Compile        |   | - Streamlit      |   | - Finalize docs  |
| - Sample amounts |   | - spaCy NLP-swap |   | - NetX Graph IF  |   |   thresholds     |   |   simulator.cpp  |   |   speedometer    |   | - README guide   |
| - Pillar 1 docx  |   | - Hold-out split |   | - Sentence-Trans |   | - Adversary Loop |   | - Socket connect |   | - Blended demo   |   | - Push to GitHub |
+------------------+   +------------------+   +------------------+   +------------------+   +------------------+   +------------------+   +------------------+
```

### Team Roles & Owners
*   **Varad (Lead/ML & C++ Systems):** Implements `simulator.cpp`, high-speed socket serialization, tabular XGBoost, manual adversarial fuzzer loop, and aggregator scoring.
*   **Swarali (Graph ML Lead):** Extracts NetworkX centralities, trains the Isolation Forest graph classifier, and builds temporal graph snapshots.
*   **Asavari (Biometric & Text ML Lead):** Engineers GMM-based biometric sampling, implements Kolmogorov-Smirnov variance tests, and builds sentence-transformer similarity checks.
*   **Shivaji (Cybersecurity Lead):** Audits the C++-to-Python endpoint security/auth protocols, details the Human-in-the-Loop review queue interface, and maps cybersecurity endpoint mappings.
*   **Gautam (Cross-Functional Support):** Standardizes data loaders, assists with Streamlit UI dashboard rendering, and aids socket communication troubleshooting.

---

## 8. Technology Stack & Repository Structure

### 8.1 Technology Stack
*   **Data Prep/Fidelity:** Python, Pandas, NumPy, spaCy (NLP/NER), Scikit-learn (GaussianMixture).
*   **ML & Detection:** XGBoost/LightGBM, imbalanced-learn (SMOTE), Scikit-learn (IsolationForest, LogisticRegression, Calibration, Metrics).
*   **Deep Learning (Embeddings & Stretch):** Sentence-Transformers (`all-MiniLM-L6-v2`), PyTorch, PyTorch Geometric (optional GraphSAGE).
*   **Systems Layer:** C++17, CMake, TCP sockets / ZeroMQ, Python FastAPI.
*   **Dashboard UI:** Streamlit.

### 8.2 Repository Layout
```
project-aegis/
├── docs/
│   └── Pillar1_Attack_Landscape.docx   # Full attack taxonomy & literature review
├── data/
│   ├── data_builder.py                  # Core synthetic generation and GMM sampling
│   └── held_out_attacks/                # Strict entity-isolated test validation set
├── cpp/
│   ├── simulator.cpp                    # High-speed compiled router
│   └── CMakeLists.txt                   # Compilation instructions
├── ml/
│   ├── blue_team_defender.py            # FastAPI orchestration backend
│   ├── graph_model.py                   # NetworkX and Isolation Forest model
│   ├── biometric_model.py               # Biometric variance and KS-testing
│   ├── text_model.py                    # Pretrained sentence transformer model
│   └── risk_aggregator.py               # Calibrated probability aggregator
├── app/
│   └── streamlit_dashboard.py           # Dashboard with real-time speedometer
└── README.md                            # Comprehensive run, build, and benchmark instructions
```
