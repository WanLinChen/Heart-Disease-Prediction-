# ECG Cardiac Disease Classification — 2023 ML Class Competition

> **Course:** Machine Learning (2023) 
> **Platform:** [Kaggle — 2023ML Competition](https://www.kaggle.com/competitions/2023mlcompetition/overview)  


## Problem Statement

Given a dataset of **12-lead ECG recordings** from 12,209 subjects, classify each patient into one of four cardiac conditions:

| Label | Description |
|-------|-------------|
| `NORM` | Normal sinus rhythm |
| `MI`   | Myocardial Infarction |
| `STTC` | ST/T-wave Change |
| `CD`   | Conduction Disturbance |

**Key constraint:** All algorithms (PCA, Naive Bayes, signal filtering) had to be **implemented from scratch** — no ML libraries (scikit-learn, etc.) were permitted. Only `numpy`, `scipy`, and `pandas` were allowed.


## Project Structure

```
├── main.py                  # Full pipeline: preprocessing → feature extraction → prediction
├── pca.py                   # Hand-crafted PCA implementation
├── naive_bayes.py           # Hand-crafted Gaussian Naive Bayes classifier
├── preprocess.py            # ECG denoising, R-peak detection, feature engineering
├── data/
│   ├── ML_Train.npy         # Training ECG signals  (12209, 12, 5000)
│   ├── ML_Train.csv         # Training metadata (age, sex, label, ...)
│   ├── ML_Test.npy          # Test ECG signals  (6000, 12, 5000)
│   └── ML_Test.csv          # Test metadata
└── README.md
```

---

## Methodology

### A. Exploratory Data Analysis

We analyzed the `.csv` metadata (age, sex, height, weight, label) via histograms before choosing features.

Key findings:
- **Age:** NORM patients are significantly younger on average. Subjects over 70 have NORM rate < 50%.
- **Sex:** Males have significantly higher rates of CD and MI — consistent with the known cardioprotective effect of estrogen.
- **Height / Weight:** No strong discriminative signal found; dropped from final feature set.

### B. ECG Signal Preprocessing

1. **Denoising:** Applied a **10th-order Butterworth low-pass filter** (cutoff 100 Hz, fs = 500 Hz) using `scipy.signal.filtfilt` to remove high-frequency noise.
2. **R-peak detection:** Two methods were evaluated:
   - *Threshold-based* (height > 0.3): Can extract multiple peaks → enables period calculation, but ~29% of subjects had no detectable peak.
   - *Maximum-value based* (final choice): Guaranteed to find exactly one peak per signal; used as a scalar feature.
3. **Feature selection — PCA:** Applied hand-crafted PCA with a **Proportion of Variance (PoV) > 0.90** criterion. This reduced 5,000 time-domain ECG samples per lead down to **397 principal components**.

### C. Feature Engineering

Final feature vector per subject (age ≥ 40 group):

| Feature | Dimension |
|---------|-----------|
| PCA of Lead 1 (397 components) | 397 |
| PCA of Lead 4 (397 components) | 397 |
| R-peak amplitude | 1 |
| Gender | 1 |
| **Total** | **796** |

Lead 1 + Lead 4 combination was found empirically to give the highest accuracy among all two-lead combinations tested.

### D. Age-Based Stratification

Rather than training a single classifier on all subjects, we split the data by age:

```
Age < 40  →  Forced prediction: NORM
             (>95% of this group are healthy; avoids classifier confusion)

Age ≥ 40  →  Gaussian Naive Bayes on full feature vector
```

This age split was selected after testing thresholds of 20, 30, 40, and 50; **age 40** yielded the best Kaggle score.

### E. Hierarchical (Layered) Classification — Explored

We also explored a cascaded approach:
```
Step 1: Healthy (NORM) vs. Diseased?
Step 2 (if diseased): STTC vs. other?
Step 3 (if not STTC): CD vs. MI?
```
This mirrored known clinical patterns (STTC more prevalent in females; CD/MI more prevalent in males), but did not outperform the single-classifier + age-split strategy.

### F. Classifier — Hand-Crafted Gaussian Naive Bayes

We implemented Gaussian Naive Bayes entirely from scratch using only NumPy. For each class $c$, the log-posterior is:

$$\log P(c \mid \mathbf{x}) = \log P(c) + \sum_{j} \left[ -\frac{1}{2}\log(2\pi\sigma_{c,j}^2) - \frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right]$$

The predicted class is the one with the highest log-posterior.


## Results

| Feature Combination | Age Split | PCA Dims | R-peak | Sex | Accuracy |
|---------------------|-----------|----------|--------|-----|----------|
| Lead 1 & 4 | Hierarchical | 397 | ✗ | ✗ | 0.3938 |
| Lead 1 & 4 | <40 / ≥40 | 397 | ✗ | ✗ | 0.4288 |
| Lead 1 & 4 | <40 / ≥40 | 397 | ✓ | ✗ | 0.4300 |
| Lead 1 & 4 | None | 68 | ✗ | ✗ | 0.3983 |
| Lead 1 & 4 | <40 / ≥40 | 397 | ✓ | ✓ | **0.4300** |
| Lead 1 & 4 | <40 / 40–80 / ≥80 | 397 | ✓ | ✓ | 0.4100 |

**Kaggle Final: Public 0.43 · Private 0.4389**


## Reflections & Lessons Learned

1. **Over-reliance on age stratification** — Forcing young patients to NORM introduced unquantified error that capped the ceiling accuracy regardless of other feature changes.
2. **Height & weight abandoned too early** — With more domain knowledge these could have been useful (e.g., BMI correlates with cardiac risk).
3. **Frequency-domain features underexplored** — Wavelet transforms (covered in signals & systems) could extract clinically relevant frequency-band features.
4. **Only 2 of 12 leads used** — Each lead captures a different cardiac axis. Using all 12 leads with individual PCA would have given the classifier substantially more information.
5. **R-peak alignment missing** — PCA on raw time-series is sensitive to temporal misalignment. Aligning signals at the R-peak before applying PCA would have improved principal component quality. Heart rate (R-R interval) is also a meaningful clinical feature that was unrecoverable with single-peak detection.


## Setup & Usage

```bash
# Install dependencies (only standard scientific Python stack)
pip install numpy scipy pandas matplotlib

# Run the full pipeline
python main.py
```

**Input files required** (place in `./data/` or adjust paths in `main.py`):
- `ML_Train.npy` — shape `(12209, 12, 5000)`
- `ML_Train.csv` — columns: `SubjectId`, `age`, `sex`, `height`, `weight`, `Label`
- `ML_Test.npy`  — shape `(6000, 12, 5000)`
- `ML_Test.csv`  — columns: `SubjectId`, `age`, `sex`, `height`, `weight`

**Output:** `output.csv` with columns `SubjectId`, `Label` (integer 0–3)


## References

- Pan-Tompkins QRS Detection: https://github.com/antimattercorrade/Pan_Tompkins_QRS_Detection  
- Gender differences in cardiac disease: https://academic.oup.com/eurheartj/article/34/41/3217/517337  
- ECG lead anatomy: https://www.nurseslearning.com/courses/nrp/nrp1619/Section%205/index.htm
