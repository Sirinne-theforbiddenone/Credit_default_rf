# Technical Specification: Multi-Class Credit Score Classification (`Credit_Score_Clean.csv`)

## 1. Overview & Objectives

- **Dataset**: `data/raw/Credit_Score_Clean.csv` (documented in `data/credit_score_clean_dataset.md`)
- **Shape**: `31,711` observations × `21` columns (`20` input features + `1` target column)
- **SHA-256**: `4a85351744d8e15f0c3285640d76ab6794a8b728583e3358c917b9bfa4a4b836`
- **Task**: 3-class credit score classification (`Credit_Score`):
  - `Poor` $\rightarrow 0$ (`4,430` rows, `13.97%`)
  - `Standard` $\rightarrow 1$ (`19,730` rows, `62.22%`)
  - `Good` $\rightarrow 2$ (`7,551` rows, `23.81%`)
- **Core Deliverables**:
  1. End-to-end Exploratory Data Analysis (EDA) and leak-free preprocessing pipeline.
  2. Training and evaluation of library baseline models: **Logistic Regression** and **Decision Tree**.
  3. Training and evaluation of **Random Forest (Scikit-Learn)** with hyperparameter and **class-weight tuning**.
  4. Implementation, training, and evaluation of a **Custom Random Forest from scratch** (pure `NumPy`, no ML library).
  5. Unified comparison of all methods focusing on **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** (Macro & Weighted One-vs-Rest).

---

## 2. Project Artifacts & File Locations

| Path | Description |
|---|---|
| `data/raw/Credit_Score_Clean.csv` | Raw input dataset (read-only) |
| `data/credit_score_clean_dataset.md` | Dataset summary and data quality reference |
| `source/credit_score_rf.ipynb` | Main self-contained Jupyter Notebook containing all stages |
| `source/logs/train_credit_score_rf_<timestamp>.txt` | Execution and training log file |
| `results/run/credit_score_model_comparison.csv` | Final evaluation metrics table across all models |
| `results/visualize/*.png` | Generated EDA, ROC, Confusion Matrix, and comparison plots |

---

## 3. Stage 1: Exploratory Data Analysis (EDA)

1. **Automated Data Integrity Checks**:
   - Assert exact shape `(31711, 21)` and presence of `"Credit_Score"`.
   - Verify `0` missing (`NaN`) values, `0` `+/-inf` values in numeric columns, and `0` full-row duplicates.
   - Verify that `Num_Credit_Inquiries` contains only whole numbers and cast from `float64` to `int64`.
2. **Target & Categorical Feature Analysis**:
   - Plot class distribution of `Credit_Score` (`Poor`, `Standard`, `Good`) with counts and percentages.
   - Plot stacked/grouped target proportions across `Credit_Mix` (`Bad`, `Standard`, `Good`), `Payment_of_Min_Amount` (`No`, `Yes`, `NM` — keeping `NM` as a distinct category), `Payment_Behaviour` (6 categories), and `Occupation` (15 categories).
3. **Numeric Feature & Outlier Analysis**:
   - Inspect and visualize right-skewed features with extreme values:
     - `Annual_Income` (median `36,993.94`, p99 `~179,166.48`, max `24,198,062.00`; `298` rows `> 1,000,000`).
     - `Num_of_Delayed_Payment` (median `11`, Q3 `15`, max `4,397`; `229` rows `> 100`).
   - Inspect domain anomalies without blind deletion:
     - `Age < 18` (`956` rows aged `14–17`).
     - `Credit_History_Age_Months == 0` (`2,863` rows, `9.03%`).
   - Compare medians and interquartile ranges (IQR) by class for key discriminative features: `Interest_Rate`, `Delay_from_due_date`, `Outstanding_Debt`, `Credit_History_Age_Months`, and `Credit_Utilization_Ratio`.
   - Generate a numeric correlation heatmap (`16` numeric features).

---

## 4. Stage 2: Leak-Free Preprocessing & Feature Engineering

### 4.1. Data Splitting
- Encode target labels consistently: `{"Poor": 0, "Standard": 1, "Good": 2}`.
- Perform a **Stratified 60 / 20 / 20 split** (`random_state=42`):
  - **Train**: `60%` (`19,026` rows) — used to fit all outlier thresholds, encoders, scalers, and models.
  - **Validation**: `20%` (`6,342` rows) — used for hyperparameter selection and class-weight / probability-weight tuning.
  - **Test**: `20%` (`6,343` rows) — held out strictly for final evaluation.

### 4.2. Outlier Handling & Domain Flags (Fit on Train Only)
- Compute upper 99th-percentile (`p99`) thresholds **strictly on `X_train`** for `Annual_Income` and `Num_of_Delayed_Payment`, and clip (`np.clip`) `X_train`, `X_val`, and `X_test` using those learned bounds.
- Retain rows with `Age < 18` and add binary indicator features:
  - `is_under_18 = (Age < 18).astype(int)`
  - `has_no_credit_history = (Credit_History_Age_Months == 0).astype(int)`

### 4.3. Feature Engineering
Create domain-specific ratios and composite features across all splits:
- `debt_to_income_ratio = Outstanding_Debt / (Annual_Income + 1e-5)`
- `annual_emi_to_income_ratio = (Total_EMI_per_month * 12.0) / (Annual_Income + 1e-5)`
- `annual_investment_to_income_ratio = (Amount_invested_monthly * 12.0) / (Annual_Income + 1e-5)`
- `balance_to_debt_ratio = Monthly_Balance / (Outstanding_Debt + 1e-5)`
- `total_financial_accounts = Num_Bank_Accounts + Num_Credit_Card + Num_of_Loan`
- `delay_severity_score = Delay_from_due_date * np.log1p(Num_of_Delayed_Payment)`
- Decompose `Payment_Behaviour` into ordinal components:
  - `spending_level`: `Low_spent` $\rightarrow 0$, `High_spent` $\rightarrow 1$
  - `payment_value_tier`: `Small_value_payments` $\rightarrow 0$, `Medium_value_payments` $\rightarrow 1$, `Large_value_payments` $\rightarrow 2$

### 4.4. Encoding & Scaling
- **Ordinal Encoding**: Map `Credit_Mix` using fixed domain order `{"Bad": 0, "Standard": 1, "Good": 2}`.
- **One-Hot Encoding**: Apply `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to nominal categorical columns (`Occupation`, `Payment_of_Min_Amount`, `Payment_Behaviour`).
- **Feature Scaling**: Fit `StandardScaler` on numeric/ordinal columns of `X_train` and transform `X_val` and `X_test` for scale-sensitive models (Logistic Regression). Tree-based models may use either the encoded unscaled matrix or the unified preprocessed matrix.

---

## 5. Stage 3: Library Models Training & Tuning (Scikit-Learn)

### 5.1. Logistic Regression
- Train multi-class `LogisticRegression(max_iter=1000, random_state=42)` on scaled features.
- Compare `class_weight=None` vs. `class_weight="balanced"` on the validation set and select the configuration maximizing validation **Macro F1**.

### 5.2. Decision Tree Classifier
- Train `DecisionTreeClassifier(random_state=42)` across candidate `max_depth` values (`[8, 12, 16, 20, None]`) and `class_weight` (`[None, "balanced"]`) on the validation set; select the best validation estimator.

### 5.3. Random Forest with Hyperparameter & Class-Weight Tuning
1. **Baseline Random Forest**:
   - `RandomForestClassifier(n_estimators=100, class_weight=None, random_state=42, n_jobs=-1)`
2. **Tuned Random Forest**:
   - Run `RandomizedSearchCV` (`cv=5` stratified, `scoring="f1_macro"`, `n_iter=20`, `random_state=42`, `n_jobs=-1`) over:
     - `n_estimators`: `[200, 300, 400]`
     - `max_depth`: `[12, 16, 20, 25, None]`
     - `min_samples_split`: `[2, 5, 10]`
     - `min_samples_leaf`: `[1, 2, 4]`
     - `max_features`: `["sqrt", "log2", 0.3]`
     - `class_weight`: `[None, "balanced", "balanced_subsample", {0: 3.0, 1: 1.0, 2: 2.0}, {0: 4.5, 1: 1.0, 2: 2.5}]`
   - **Post-Training Class-Weight / Threshold Multiplier Optimization**:
     - On the validation set, optimize class probability multipliers $\mathbf{w} = (w_0, 1.0, w_2)$ where $\hat{y} = \arg\max_{c \in \{0,1,2\}} (w_c \cdot P(y=c \mid \mathbf{x}))$ to maximize validation **Macro F1**, then apply $\mathbf{w}^*$ to test predictions.

---

## 6. Stage 4: Custom Random Forest From Scratch (Pure `NumPy`)

Implement a clean, self-contained Random Forest without using `scikit-learn` tree/ensemble classes:

### 6.1. `DecisionTreeScratch`
- **Splitting Criterion**: Multi-class Weighted Gini Impurity for $K = 3$ classes:
  $$\text{Gini}(S) = 1 - \sum_{c=0}^{K-1} \left( \frac{\sum_{i \in S, y_i = c} w_i}{\sum_{i \in S} w_i} \right)^2$$
- **Fast Candidate Threshold Selection**:
  - At each internal node, randomly sample `max_features` ($\lfloor\sqrt{d}\rfloor$) feature indices without replacement.
  - For each selected feature, evaluate up to `n_bins = 32` quantile thresholds (`np.quantile` / unique values if $\le 32$) using vectorized NumPy masks.
- **Stopping Rules**:
  - `depth >= max_depth`, `n_samples < min_samples_split`, pure node (`Gini == 0`), or zero impurity gain.
- **Leaf Representation**:
  - Store the normalized weighted class probability vector $[P(y=0), P(y=1), P(y=2)]$ of shape `(3,)`.

### 6.2. `RandomForestScratch`
- **Hyperparameters**: `n_estimators=35`, `max_depth=12`, `min_samples_split=10`, `max_features="sqrt"`, `class_weight="balanced"`, `n_bins=32`, `random_state=42`.
- **Training (`fit`)**:
  - Compute per-class weights $w_c = \frac{N}{K \cdot N_c}$ when `class_weight="balanced"` (or accept a custom dict).
  - For each tree $t \in \{1, \dots, T\}$, draw a bootstrap sample of size $N$ with replacement and fit a `DecisionTreeScratch` instance.
- **Inference (`predict_proba` & `predict`)**:
  - `predict_proba(X)`: Average the `(n_samples, 3)` probability matrices returned by all trees.
  - `predict(X)`: Return `np.argmax(predict_proba(X), axis=1)`.
- **Verification**:
  - Include `assert` checks verifying `predict_proba(X_test)` has shape `(len(X_test), 3)`, row sums equal `1.0` (`np.allclose`), and Macro F1 beats a dummy baseline.

---

## 7. Stage 5: Evaluation & Model Comparison

Evaluate all models on the untouched **Test Set (`6,343` rows)**:
1. **Logistic Regression (Tuned Class Weight)**
2. **Decision Tree (Scikit-Learn)**
3. **Random Forest Baseline (Scikit-Learn, Unweighted)**
4. **Random Forest Tuned (Scikit-Learn, Hyperparameter + Class-Weight Tuned)**
5. **Random Forest From Scratch (Pure `NumPy`, Class-Weighted)**

### 7.1. Required Metrics
- **Primary Multi-Class Metrics**:
  - `Macro Precision`, `Macro Recall`, `Macro F1-Score`
  - `Weighted Precision`, `Weighted Recall`, `Weighted F1-Score`
  - `Macro ROC-AUC (OvR)` and `Weighted ROC-AUC (OvR)` via `roc_auc_score(y_test, y_prob, multi_class="ovr", average=...)`
- **Per-Class Breakdown**:
  - Precision, Recall, and F1-Score for `Poor (0)`, `Standard (1)`, and `Good (2)`
- **Computational Efficiency**:
  - Fit time (seconds) and inference time (seconds) per model

### 7.2. Required Visualizations & Outputs
- Save comparison table to `results/run/credit_score_model_comparison.csv`.
- Save visualizations to `results/visualize/`:
  1. `eda_target_and_categorical.png`: Target distribution and categorical breakdowns.
  2. `eda_numeric_outliers_and_corr.png`: Outlier boxplots/histograms and correlation heatmap.
  3. `model_metrics_comparison.png`: Grouped bar chart comparing Macro Precision, Macro Recall, Macro F1, and Macro ROC-AUC across all models.
  4. `multiclass_roc_curves.png`: One-vs-Rest ROC curves comparing the models.
  5. `confusion_matrices_comparison.png`: Side-by-side $3 \times 3$ confusion matrices for all models.
  6. `rf_feature_importance.png`: Top 15 most important features from the best Random Forest model.

---

## 8. Coding & Documentation Standards

1. **Function & Method Comment Blocks**:
   Every function and class method must have the exact header block immediately above `def`:
   ```python
   # ******************************************************************
   # Purpose: <Describe what the function does and why it is needed.>
   # Inputs: <parameter_name>: <Describe the input and its role.>
   # Returns: <Describe the returned value and its meaning.>
   # Author: Sirinne
   # last modified:
   # ******************************************************************
   ```
2. **Naming Conventions**:
   - Functions, methods, and variables: `snake_case`
   - Classes: `PascalCase` (`DecisionTreeScratch`, `RandomForestScratch`)
   - Constants: `UPPER_SNAKE_CASE` (`RANDOM_SEED`, `TARGET_COL`, `CLASS_MAPPING`)
3. **Language**:
   - Code comments and function headers: **English**
   - Notebook Markdown explanations, analysis, and conclusions: **Vietnamese**
