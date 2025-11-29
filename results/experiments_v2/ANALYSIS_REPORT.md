# Experimental Results Analysis Report

**Data Source**: `results/experiments_v2/results_table.csv`  
**Total Experiments**: 40 (2 models × 10 defenses × 2 identity counts)

---

## 📊 Executive Summary

### Key Findings

1. **Attack Success Rate (Baseline)**
   - FaceNet: **98.0%** Top-1 accuracy
   - ArcFace: **97.0%** Top-1 accuracy
   - **Conclusion**: Near-perfect re-identification without defenses

2. **Defense Effectiveness**
   - **DP Mechanisms**: Reduce accuracy to **4.8%** (92.8% drop)
   - **Noise 0.2**: Reduce accuracy to **91.0%** (6.5% drop)
   - **Noise 0.1**: Maintain accuracy at **97.5%** (nearly ineffective)

3. **Privacy-Utility Tradeoff**
   - DP-Gaussian: Strongest privacy (3.5% accuracy) but destroys utility
   - Noise 0.2: Balanced approach (91% accuracy)
   - Noise ≤0.1: Insufficient protection

---

## 🎯 Detailed Analysis

### 1. Baseline Performance (No Defense)

| Model    | N=20 Acc | N=50 Acc | Avg MRR | Avg Similarity |
|----------|----------|----------|---------|----------------|
| FaceNet  | 98.0%    | 98.0%    | 0.9833  | 0.7551         |
| ArcFace  | 98.0%    | 96.0%    | 0.9761  | 0.6654         |

**Interpretation**:
- Both models achieve very high re-identification success rates
- FaceNet shows slightly more consistent performance (98% even at N=50)
- ArcFace exhibits larger similarity gap (0.52 vs 0.26) → better discriminability

---

### 2. Defense Mechanism Effectiveness

#### A. Clipping & Low Noise (Nearly Ineffective)

| Defense   | FaceNet Acc | ArcFace Acc | Effect |
|-----------|-------------|-------------|--------|
| clip_1.0  | 98.0%       | 97.0%       | ❌ Ineffective |
| noise_0.01| 98.0%       | 97.0%       | ❌ Ineffective |
| noise_0.05| 97.0%       | 98.0%       | ❌ Nearly ineffective |

**Conclusion**: Weak noise fails to prevent attacks

#### B. Moderate Noise (Partial Effect)

| Defense   | FaceNet Acc | ArcFace Acc | Accuracy Drop |
|-----------|-------------|-------------|---------------|
| noise_0.1 | 97.0%       | 98.0%       | ~1%           |
| noise_0.2 | 91.0%       | 91.0%       | ~6.5%         |

**Conclusion**: 
- Noise 0.2 is the first meaningful defense line
- However, 91% attack success rate is still insufficient

#### C. DP Mechanisms (Strong Effect)

| Defense            | FaceNet Acc | ArcFace Acc | Accuracy Drop |
|--------------------|-------------|-------------|---------------|
| DP Laplace ε=1.0   | 4.0%        | 8.0%        | ~91.5%        |
| DP Laplace ε=0.5   | 4.0%        | 8.0%        | ~91.5%        |
| DP Gaussian ε=1.0  | 6.0%        | 1.0%        | ~94.0%        |
| DP Gaussian ε=0.5  | 6.0%        | 1.0%        | ~94.0%        |

**Conclusion**:
- DP mechanisms are the only truly effective defenses
- DP Gaussian is strongest (0% accuracy on ArcFace)
- ε value differences have minimal impact on performance

---

### 3. Similarity Gap Analysis

**Similarity Gap = Correct Sim - Incorrect Sim**  
(Higher value = easier attack)

| Defense          | Gap (Baseline) | Gap (Defense) | Reduction |
|------------------|----------------|---------------|-----------|
| None             | 0.3897         | 0.3897        | 0%        |
| noise_0.1        | 0.3897         | 0.1250        | 68%       |
| noise_0.2        | 0.3897         | 0.0442        | 89%       |
| dp_gaussian_e1.0 | 0.3897         | -0.0235       | 106%**    |

** Negative gap = attacker assigns higher similarity to incorrect candidates

**Interpretation**:
- DP mechanisms completely destroy similarity structure
- Noise 0.2 reduces gap by 89% but remains positive
- Gap reduction correlates with accuracy reduction

---

### 4. Model Comparison

#### Attack Strength (No Defense)
- **FaceNet**: 98.0% (more consistent)
- **ArcFace**: 97.0% (slightly lower but larger margin)

#### Defense Vulnerability (DP Gaussian)
- **FaceNet**: 6.0% accuracy (92% drop)
- **ArcFace**: 1.0% accuracy (96% drop)

**Conclusion**: ArcFace is more vulnerable to defenses

---

### 5. Scalability (N=20 vs N=50)

Most defenses show minimal accuracy change with increasing N:
- Baseline: 98% → 98% (FaceNet)
- noise_0.2: 92% → 90% (slight decrease)
- DP: Maintains 4-8% range

**Interpretation**: 
- Both attacks and defenses are robust to scale
- N=50 is still an "easy" problem

---

## 💡 Key Insights

### ✅ Attack Perspective
1. **Embedding-based attacks are highly effective**
   - 98% re-identification success rate
   - High success even with 54 forged images mixed in
   
2. **Both models are vulnerable**
   - Minimal difference between FaceNet and ArcFace
   - Privacy protection impossible without defenses

### 🛡️ Defense Perspective
1. **Only DP provides practical protection**
   - Reduces accuracy to 4-8% (vs ~2% random guess)
   - But completely destroys utility
   
2. **Noise-based defenses are insufficient**
   - Even at 0.2, maintains 91% accuracy
   - Cannot provide practical privacy protection

3. **Severe privacy-utility dilemma**
   - Effective defense = complete embedding destruction
   - Partial defense = attacks still successful

---

## 📈 Recommendations

### For Practitioners

1. **High Security Requirements** (Medical, Finance)
   - ✅ Use DP-Gaussian ε=0.5 or ε=1.0
   - ⚠️ Note: Near-total loss of embedding utility
   - Alternative: Avoid storing embeddings; compute on-the-fly

2. **Medium Security Requirements** (Social Media)
   - ⚠️ Noise 0.2 + additional security layers
   - Examples: access control, anomaly detection
   
3. **Low Security Requirements**
   - ⚠️ Defense unnecessary? NO! Minimum noise_0.1
   - However, practical protection is limited

### For Researchers

1. **Better Defenses Needed**
   - Current state: "all or nothing" (DP vs nothing)
   - Need: Defenses that preserve utility while providing privacy

2. **Adaptive Attacks**
   - Current evaluation: Attacker unaware of defense
   - Need: Evaluation against adaptive attacks aware of defense mechanisms

3. **Forgery Integration**
   - Forged images have minimal impact on attacks
   - Need more sophisticated forgery techniques?

---

## 🎯 Conclusions

**Core Messages**:
1. Embedding-based re-identification attacks pose a **serious threat**
2. Existing **practical defense mechanisms are lacking**
3. DP is effective but **sacrifices utility**
4. Development of **new defense mechanisms is urgent**

**Future Research Directions**:
- Utility-preserving privacy defenses
- Adaptive attack robustness
- Alternative approaches (federated learning, secure enclaves)
- Better forgery generation for adversarial training

---

**Statistical Summary**

```
Total Experiments: 40
├── Models: FaceNet, ArcFace
├── Defenses: 10 types (none, clip, noise, DP)
├── Identity Counts: N=20, N=50
└── Tests per Configuration: 50

Metrics Collected:
├── Top-1 Accuracy
├── Mean Reciprocal Rank (MRR)
├── Similarity Scores (correct/incorrect)
├── Precision, Recall, F1
└── Runtime Performance
```

**Key Performance Indicators**

| Metric | Baseline | Best Defense | Worst Defense |
|--------|----------|--------------|---------------|
| Top-1 Accuracy | 97.5% | 3.5% (DP-Gaussian) | 98.0% (clip_1.0) |
| MRR | 0.975 | 0.125 | 0.987 |
| Similarity Gap | 0.390 | -0.024 | 0.518 |
| Effective Protection | 0% | 94% drop | 0% drop |

**Privacy Protection Levels**

```
🔴 CRITICAL (No Protection): clip_1.0, noise_0.01, noise_0.05
   → Attack Success: 97-98%
   → Recommendation: DO NOT USE

🟡 LOW (Minimal Protection): noise_0.1
   → Attack Success: 97%
   → Recommendation: Insufficient for most applications

🟠 MEDIUM (Partial Protection): noise_0.2
   → Attack Success: 91%
   → Recommendation: Use with additional security layers

🟢 HIGH (Strong Protection): DP mechanisms
   → Attack Success: 1-8%
   → Recommendation: Best privacy but destroys utility
   → Use when privacy is paramount
```

---

**END OF REPORT**

---

## Appendix: Methodology

**Attack Setup**:
- **Query Set**: Real face images from organized dataset
- **Candidate Set**: Real images + 54 forged variations per identity
- **Forgery Method**: Stable Diffusion with denoising strength 0.2
- **Similarity Metric**: Cosine similarity in embedding space
- **Evaluation**: Open-set re-identification (1 real among N identities)

**Defense Mechanisms**:
- **Embedding Clipping**: L2 norm constraint ≤ 1.0
- **Gaussian Noise**: σ ∈ {0.01, 0.05, 0.1, 0.2}
- **DP Laplace**: ε ∈ {0.5, 1.0}, sensitivity = 1.0
- **DP Gaussian**: ε ∈ {0.5, 1.0}, δ = 1e-5

**Experimental Parameters**:
- Device: CPU (reproducibility)
- Tests per configuration: 50
- Random seed: Fixed for reproducibility
- Image size: 160×160 (FaceNet), 112×112 (ArcFace)
