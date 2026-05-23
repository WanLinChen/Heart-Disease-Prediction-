#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
-------
ECG Cardiac Disease Classification — 2023 ML Class Competition (Team 9)

Task:
    Classify 12-lead ECG recordings into one of four cardiac conditions:
        0: NORM  (Normal)
        1: MI    (Myocardial Infarction)
        2: STTC  (ST/T-wave Change)
        3: CD    (Conduction Disturbance)

Constraint:
    All ML algorithms must be hand-crafted from scratch.
    Only numpy, scipy, and pandas are permitted.

Pipeline:
    1. Load ECG signals and metadata
    2. Denoise signals (Butterworth low-pass filter)
    3. Extract R-peak amplitude from each denoised signal
    4. Apply hand-crafted PCA on Lead 1 and Lead 4
    5. Age-based stratification: under-40s → force NORM; over-40s → Naive Bayes
    6. Combine predictions and export to CSV
"""

import os
import numpy as np
import scipy.signal as sig
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR       = './data/'
TRAIN_NPY      = os.path.join(DATA_DIR, 'ML_Train.npy')
TRAIN_CSV      = os.path.join(DATA_DIR, 'ML_Train.csv')
TEST_NPY       = os.path.join(DATA_DIR, 'ML_Test.npy')
TEST_CSV       = os.path.join(DATA_DIR, 'ML_Test.csv')
OUTPUT_CSV     = 'output.csv'

FS             = 500          # Sampling frequency (Hz)
LOWCUT         = 100          # Low-pass filter cutoff (Hz)
FILTER_ORDER   = 10           # Butterworth filter order
N_SAMPLES      = 5000         # ECG time-points used per lead
PCA_COMPONENTS = 397          # Number of PCs (PoV > 0.90 criterion)
AGE_THRESHOLD  = 40           # Subjects below this age → forced NORM

# Label mapping (string → int)
LABEL_MAP = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3}


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted PCA
# ─────────────────────────────────────────────────────────────────────────────

def pca(X: np.ndarray, n_components: int) -> tuple:
    """
    Principal Component Analysis — implemented from scratch.

    Centres the data, computes the covariance matrix, and projects onto the
    top `n_components` eigenvectors (sorted by descending eigenvalue).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    X_projected : np.ndarray, shape (n_samples, n_components)
        Data projected into the PCA subspace.
    eigenvalues : np.ndarray, shape (n_features,)
        All eigenvalues of the covariance matrix (sorted descending).
    eigenvectors : np.ndarray, shape (n_features, n_features)
        Corresponding eigenvectors (columns), sorted to match eigenvalues.
    """
    # Centre the data (zero mean per feature)
    X_centred = X - np.mean(X, axis=0)

    # Covariance matrix: shape (n_features, n_features)
    cov = np.cov(X_centred, rowvar=False)

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by descending eigenvalue (most variance first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Project data onto top n_components principal components
    X_projected = X_centred @ eigenvectors[:, :n_components]

    return X_projected, eigenvalues, eigenvectors


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted Gaussian Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNB:
    """
    Gaussian Naive Bayes classifier — implemented from scratch.

    Assumption: features are conditionally independent given the class label,
    and each feature follows a Gaussian distribution within each class.

    For each class c, the log-posterior is computed as:

        log P(c | x) = log P(c)
                       + Σ_j [ -0.5 * log(2π σ²_cj)
                                - 0.5 * (x_j - μ_cj)² / σ²_cj ]

    Prediction: argmax over classes.
    """

    def __init__(self):
        self.class_prior_  = None   # shape (n_classes,)
        self.mu_           = None   # shape (n_classes, n_features)
        self.sigma_        = None   # shape (n_classes, n_features)  — variance

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        """
        Estimate class priors and per-class Gaussian parameters from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — integer class labels

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        classes   = np.unique(y)
        n_classes = len(classes)

        self.class_prior_ = np.zeros(n_classes)
        self.mu_          = np.zeros((n_classes, n_features))
        self.sigma_       = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_prior_[i] = X_c.shape[0] / n_samples   # P(c)
            self.mu_[i, :]       = np.mean(X_c, axis=0)       # μ_c
            self.sigma_[i, :]    = np.var(X_c, axis=0)        # σ²_c

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each sample using the MAP rule.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)  — predicted integer labels
        """
        n_samples  = X.shape[0]
        n_classes  = len(self.class_prior_)

        log_posteriors = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # Log-likelihood under Gaussian assumption (summed over features)
            log_likelihood = np.sum(
                -0.5 * np.log(2 * np.pi * self.sigma_[i, :])
                - 0.5 * ((X - self.mu_[i, :]) ** 2) / self.sigma_[i, :],
                axis=1
            )
            log_posteriors[:, i] = log_likelihood + np.log(self.class_prior_[i])

        return np.argmax(log_posteriors, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Signal Processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def denoise_signals(signals: np.ndarray,
                    fs: int = FS,
                    lowcut: float = LOWCUT,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter to remove high-frequency noise.

    Uses scipy.signal.filtfilt (forward-backward pass) to achieve zero phase
    distortion, which preserves the temporal structure of ECG waveforms.

    Parameters
    ----------
    signals : np.ndarray, shape (n_subjects, n_samples)
    fs      : int    — sampling frequency in Hz
    lowcut  : float  — cutoff frequency in Hz
    order   : int    — filter order

    Returns
    -------
    denoised : np.ndarray, same shape as input
    """
    b, a = sig.butter(order, lowcut, btype='lowpass', fs=fs)
    return sig.filtfilt(b, a, signals)


def extract_rpeak_amplitude(signals: np.ndarray) -> np.ndarray:
    """
    Detect the R-peak amplitude in each ECG signal using the global maximum.

    Note: This is the "maximum-value" approach (Method 2 from our report).
    While it cannot extract R-R intervals, it guarantees a valid detection
    for every subject — including those with low-amplitude R-waves where
    threshold-based methods fail (~29% failure rate in our dataset).

    Parameters
    ----------
    signals : np.ndarray, shape (n_subjects, n_samples)

    Returns
    -------
    rpeaks : np.ndarray, shape (n_subjects, 1)  — amplitude of each R-peak
    """
    rpeaks = np.max(signals, axis=1).reshape(-1, 1)
    return rpeaks


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(label_series: pd.Series) -> np.ndarray:
    """Map string disease labels to integer class indices."""
    return np.array([LABEL_MAP[l] for l in label_series]).reshape(-1, 1)


def load_and_extract_features(npy_path: str,
                               csv_path: str,
                               n_subjects: int,
                               is_train: bool = True) -> dict:
    """
    Full feature extraction pipeline for one split (train or test).

    Steps:
        1. Load raw ECG array and metadata CSV
        2. Extract Lead 1 and Lead 4 time-series
        3. Denoise both leads
        4. Extract R-peak amplitude from Lead 1
        5. Apply PCA to Lead 1 and Lead 4 separately
        6. Build combined feature matrix

    Parameters
    ----------
    npy_path   : str  — path to .npy ECG file
    csv_path   : str  — path to .csv metadata file
    n_subjects : int  — number of subjects in this split
    is_train   : bool — if True, also extract and return labels

    Returns
    -------
    dict with keys:
        'features'  : np.ndarray, shape (n_subjects, n_features)
        'age'       : np.ndarray, shape (n_subjects,)
        'subject_ids': list
        'labels'    : np.ndarray (only if is_train=True)
    """
    print(f'\nLoading data from {npy_path} ...')
    ecg    = np.load(npy_path)           # (n_subjects, 12, 5000)
    info   = pd.read_csv(csv_path)

    # ── Lead extraction ──────────────────────────────────────────────────────
    lead1 = ecg[:, 0, :N_SAMPLES].astype(float)   # Lead I
    lead4 = ecg[:, 3, :N_SAMPLES].astype(float)   # Lead aVR (index 3)

    # ── Denoising ────────────────────────────────────────────────────────────
    print('Denoising Lead 1 and Lead 4 ...')
    lead1_clean = denoise_signals(lead1)
    lead4_clean = denoise_signals(lead4)

    # ── R-peak amplitude (from denoised Lead 1) ───────────────────────────────
    print('Extracting R-peak amplitudes ...')
    rpeak = extract_rpeak_amplitude(lead1_clean)   # (n_subjects, 1)

    # ── PCA on each lead ─────────────────────────────────────────────────────
    print(f'Applying PCA (retaining {PCA_COMPONENTS} components) ...')
    pca1, _, _ = pca(lead1_clean, PCA_COMPONENTS)  # (n_subjects, 397)
    pca4, _, _ = pca(lead4_clean, PCA_COMPONENTS)  # (n_subjects, 397)

    # ── Demographic features ─────────────────────────────────────────────────
    sex = info['sex'].values.reshape(-1, 1).astype(float)
    age = info['age'].values

    # ── Combine into feature matrix ───────────────────────────────────────────
    # Shape: (n_subjects, 397 + 397 + 1 + 1) = (n_subjects, 796)
    features = np.concatenate([pca1, pca4, rpeak, sex], axis=1)

    result = {
        'features'    : features,
        'age'         : age,
        'subject_ids' : info['SubjectId'].tolist(),
    }

    if is_train:
        result['labels'] = encode_labels(info['Label']).flatten()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Age-Based Stratification & Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_age_split(train: dict, test: dict) -> np.ndarray:
    """
    Age-stratified classification strategy:

        Age < AGE_THRESHOLD  →  forced prediction = NORM (class 0)
        Age ≥ AGE_THRESHOLD  →  Gaussian Naive Bayes trained on same-age group

    This exploits the strong empirical observation that subjects under 40
    are overwhelmingly healthy (NORM rate > 95%), reducing classifier confusion
    at the cost of occasional false-negative disease cases in young patients.

    Parameters
    ----------
    train : dict  — output of load_and_extract_features (training split)
    test  : dict  — output of load_and_extract_features (test split)

    Returns
    -------
    predictions : np.ndarray, shape (n_test_subjects,)
        Predicted class indices (0=NORM, 1=MI, 2=STTC, 3=CD)
        Ordered to match the original test subject ordering.
    """
    # ── Training: select subjects aged ≥ 40 ──────────────────────────────────
    train_mask_ab40 = train['age'] >= AGE_THRESHOLD
    X_train_ab40   = train['features'][train_mask_ab40]
    y_train_ab40   = train['labels'][train_mask_ab40]

    # Shuffle training data to avoid any ordering bias
    perm = np.random.permutation(len(X_train_ab40))
    X_train_ab40 = X_train_ab40[perm]
    y_train_ab40 = y_train_ab40[perm]

    print(f'\nTraining Gaussian NB on {len(X_train_ab40)} subjects (age ≥ {AGE_THRESHOLD}) ...')
    gnb = GaussianNB()
    gnb.fit(X_train_ab40, y_train_ab40)

    # ── Test: split by age ────────────────────────────────────────────────────
    test_age        = test['age']
    test_subject_ids = np.array(test['subject_ids'])

    mask_ab40 = test_age >= AGE_THRESHOLD
    mask_un40 = test_age <  AGE_THRESHOLD

    ids_ab40 = test_subject_ids[mask_ab40]
    ids_un40 = test_subject_ids[mask_un40]

    print(f'Test subjects age ≥ {AGE_THRESHOLD}: {mask_ab40.sum()}  |  '
          f'age < {AGE_THRESHOLD} (forced NORM): {mask_un40.sum()}')

    # Predict for age ≥ 40 group
    X_test_ab40   = test['features'][mask_ab40]
    preds_ab40    = gnb.predict(X_test_ab40)

    # Force NORM (0) for under-40 group
    preds_un40    = np.zeros(mask_un40.sum(), dtype=int)

    # ── Merge and sort by SubjectId ───────────────────────────────────────────
    combined_ids   = np.concatenate([ids_ab40,   ids_un40])
    combined_preds = np.concatenate([preds_ab40, preds_un40])

    sort_order = np.argsort(combined_ids)
    sorted_ids   = combined_ids[sort_order]
    sorted_preds = combined_preds[sort_order]

    return sorted_ids, sorted_preds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # ── Load & extract features ───────────────────────────────────────────────
    train = load_and_extract_features(TRAIN_NPY, TRAIN_CSV,
                                      n_subjects=12209, is_train=True)
    test  = load_and_extract_features(TEST_NPY,  TEST_CSV,
                                      n_subjects=6000,  is_train=False)

    # ── Predict ───────────────────────────────────────────────────────────────
    subject_ids, predictions = predict_with_age_split(train, test)

    # ── Save output ───────────────────────────────────────────────────────────
    output_df = pd.DataFrame({
        'SubjectId': subject_ids.astype(int),
        'Label'    : predictions.astype(int),
    })
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nPredictions saved to {OUTPUT_CSV}  ({len(output_df)} rows)')

    # ── Quick sanity check: class distribution ────────────────────────────────
    dist = output_df['Label'].value_counts().sort_index()
    label_names = {v: k for k, v in LABEL_MAP.items()}
    print('\nPredicted class distribution:')
    for cls, count in dist.items():
        print(f'  {label_names[cls]:4s} ({cls}): {count:5d}  '
              f'({100 * count / len(output_df):.1f}%)')


if __name__ == '__main__':
    main()