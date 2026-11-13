# LiverAI — Liver Condition Analysis

A recreated, runnable academic prototype based on the uploaded research paper *Artificial Intelligence Frameworks for Liver Condition Analysis*.

## Features
- UCI Indian Liver Patient Dataset (ILPD; 583 records, 10 predictor features)
- Missing-value imputation and numeric scaling / gender encoding
- Logistic Regression, Random Forest, SVM, Gradient Boosting
- Stacked ensemble combining the four model pipelines
- Accuracy, precision, recall, F1, ROC-AUC and confusion matrix
- Explainable AI-style feature ranking using permutation importance
- Streamlit web interface for interactive data entry and predictions

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this ZIP folder.
3. Open Command Prompt inside the extracted `liver_ai_project` folder.
4. Create and activate a virtual environment (recommended):

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install dependencies:

   ```bat
   pip install -r requirements.txt
   ```
6. Download dataset once (requires internet):

   ```bat
   python download_dataset.py
   ```
7. Start the application:

   ```bat
   streamlit run app.py
   ```

The app opens in your browser, usually at http://localhost:8501. It can also attempt to fetch the dataset automatically on first launch.

## Dataset and label handling
The project uses UCI ILPD, which contains 583 records. The target is mapped as `Selector = 1` → liver-patient class and `Selector = 2` → non-liver-patient class, following the UCI dataset description. Dataset source and citation: Bendi Ramana & N. Venkateswarlu, *ILPD (Indian Liver Patient Dataset)*, UCI Machine Learning Repository, https://doi.org/10.24432/C5D02C (CC BY 4.0).

## Research-paper alignment and differences
The implementation includes the paper's tabular-data workflow and its Logistic Regression, Random Forest, SVM and Gradient Boosting model families. It adds a stacked ensemble and permutation-based feature explanations. It does **not** reproduce the exact historical experiment or guarantee the accuracy figures printed in the paper. Metrics are generated from a reproducible stratified 80/20 holdout split. Results can vary with library versions and data preprocessing.

The ILPD dataset is tabular. The paper's suggested CNN/image fusion, LSTM time-series analysis and federated learning require additional image, longitudinal or multi-institutional datasets and are not represented as completed features here.

## Important medical limitation
This is an educational research demonstration, not a validated medical device or diagnostic system. Predictions must not be used to diagnose, rule out disease, or guide treatment. A qualified healthcare professional must interpret laboratory results.
