# AMR Prediction Analysis

This project analyzes antimicrobial resistance (AMR) data using machine learning models to predict resistance to Azithromycin and Ciprofloxacin.

## Project Structure

```
AMR-Prediction/
├── data/
│   ├── metadata_remane_original.csv
│   ├── azm_sr_gwas_filtered_unitigs.Rtab
│   └── cip_sr_gwas_filtered_unitigs.Rtab
├── model/
│   └── model_analysis.py
└── requirements.txt
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analysis script:
```bash
cd model
python model_analysis.py
```

The script will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Train and evaluate multiple machine learning models
4. Generate visualizations and performance metrics

## Output

The script generates several output files:
- `azm_resistance_distribution.png`: Distribution of Azithromycin resistance
- `geographic_distribution.png`: Geographic distribution of resistance
- `confusion_matrix_*.png`: Confusion matrices for each model and antibiotic
- Console output with detailed performance metrics

## Models

The analysis includes three machine learning models:
1. Random Forest
2. Support Vector Machine (SVM)
3. Neural Network

Each model is evaluated using:
- Accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix 