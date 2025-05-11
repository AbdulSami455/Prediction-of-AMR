# Antimicrobial Resistance (AMR) Prediction Analysis

## Overview
This project analyzes antimicrobial resistance (AMR) data using machine learning models to predict resistance patterns across multiple antibiotics. The analysis focuses on understanding resistance patterns, geographical distribution, and developing predictive models for clinical decision support.

## Project Structure
```
AMR-Prediction/
├── data/
│   ├── metadata_remane_original.csv    # Sample metadata and resistance information
│   └── azm_sr_gwas_filtered_unitigs.Rtab  # Genetic data for Azithromycin resistance and others 
├── src/
│   ├── data_processing/               # Data loading and preprocessing
│   ├── visualization/                 # Plotting and visualization functions
│   ├── models/                        # Model training and evaluation
│   └── analysis/                      # Various analysis modules
│       ├── by_continent/             # Continent-specific analysis
│       ├── by_antibiotic/            # Antibiotic-specific analysis
│       ├── metrics/                  # Model performance metrics
│       └── clinical/                 # Clinical impact analysis
└── Output/                           # Generated analysis results
    ├── continent_analysis/           # Continent-specific results
    ├── antibiotic_analysis/          # Antibiotic-specific results
    ├── model_metrics/                # Model performance metrics
    ├── plots/                        # Generated visualizations
    ├── maps/                         # Geographical distribution maps
    └── clinical_analysis/            # Clinical impact analysis results
```

## Antibiotics Analyzed
The project analyzes resistance patterns for the following antibiotics:

1. **Azithromycin (azm_sr)**
   - Class: Macrolide
   - Common uses:
     * Respiratory infections (pneumonia, bronchitis)
     * Sexually transmitted infections (chlamydia, gonorrhea)
     * Skin and soft tissue infections
     * Ear infections (otitis media)
     * Traveler's diarrhea
   - Clinical significance: First-line treatment for many infections
   - Mechanism: Inhibits bacterial protein synthesis
   - Spectrum: Broad-spectrum, effective against Gram-positive and some Gram-negative bacteria

2. **Ciprofloxacin (cip_sr)**
   - Class: Fluoroquinolone
   - Common uses:
     * Urinary tract infections (UTIs)
     * Respiratory infections (pneumonia, bronchitis)
     * Gastrointestinal infections
     * Bone and joint infections
     * Skin and soft tissue infections
     * Anthrax exposure
   - Clinical significance: Broad-spectrum antibiotic
   - Mechanism: Inhibits DNA gyrase and topoisomerase IV
   - Spectrum: Broad-spectrum, effective against Gram-negative and some Gram-positive bacteria

3. **Ceftriaxone (cro_sr)**
   - Class: Third-generation cephalosporin
   - Common uses:
     * Severe bacterial infections
     * Meningitis
     * Gonorrhea
     * Lyme disease
     * Hospital-acquired pneumonia
     * Complicated UTIs
     * Bone and joint infections
   - Clinical significance: Critical for treating serious infections
   - Mechanism: Inhibits cell wall synthesis
   - Spectrum: Broad-spectrum, effective against Gram-negative and some Gram-positive bacteria

4. **Cefixime (cfx_sr)**
   - Class: Third-generation cephalosporin
   - Common uses:
     * Respiratory infections (pneumonia, bronchitis)
     * Urinary tract infections
     * Otitis media
     * Gonorrhea
     * Uncomplicated UTIs
     * Pharyngitis and tonsillitis
   - Clinical significance: Oral alternative to ceftriaxone
   - Mechanism: Inhibits cell wall synthesis
   - Spectrum: Broad-spectrum, effective against Gram-negative and some Gram-positive bacteria

5. **Tetracycline (tet_sr)**
   - Class: Tetracycline
   - Common uses:
     * Acne vulgaris
     * Chlamydia infections
     * Lyme disease
     * Rocky Mountain spotted fever
     * Cholera
     * Brucellosis
     * Helicobacter pylori infections
     * Periodontal disease
   - Clinical significance: Broad-spectrum antibiotic
   - Mechanism: Inhibits protein synthesis
   - Spectrum: Broad-spectrum, effective against Gram-positive and Gram-negative bacteria, rickettsiae, and some protozoa

6. **Penicillin (pen_sr)**
   - Class: Beta-lactam
   - Common uses:
     * Streptococcal infections
     * Syphilis
     * Pneumococcal infections
     * Meningococcal infections
     * Endocarditis
     * Dental infections
     * Skin and soft tissue infections
     * Rheumatic fever prophylaxis
   - Clinical significance: One of the most widely used antibiotics
   - Mechanism: Inhibits cell wall synthesis
   - Spectrum: Primarily effective against Gram-positive bacteria

### Resistance Patterns
Each antibiotic's resistance is analyzed for:
- Geographical distribution
- Temporal trends
- Cross-resistance with other antibiotics
- Clinical impact of resistance
- Genetic markers associated with resistance

### Clinical Implications
- Resistance to these antibiotics can lead to:
  * Treatment failures
  * Increased healthcare costs
  * Longer hospital stays
  * Higher mortality rates
  * Limited treatment options
  * Need for more expensive or toxic alternatives

## Analysis Components

### 1. Data Analysis
- **Geographical Distribution**
  - Interactive maps showing resistance patterns
  - Continent-specific analysis
  - Temporal trends in resistance

- **Antibiotic Resistance Patterns**
  - Distribution of sensitive vs. resistant cases
  - Cross-resistance analysis
  - Temporal trends in resistance

### 2. Machine Learning Models
- **Random Forest**
  - Handles non-linear relationships
  - Provides feature importance
  - Robust to overfitting

- **Support Vector Machine (SVM)**
  - Effective in high-dimensional spaces
  - Good for binary classification
  - Handles non-linear boundaries

- **Gradient Boosting**
  - High predictive accuracy
  - Handles mixed data types
  - Provides feature importance

### 3. Model Evaluation
- **Performance Metrics**
  - Accuracy and balanced accuracy
  - Precision, recall, and F1-score
  - ROC curves and AUC scores
  - Confusion matrices

- **Learning Curves**
  - Model performance vs. training size
  - Bias-variance analysis
  - Overfitting detection

### 4. Clinical Impact Analysis
- **Prediction Impact**
  - Correctly identified resistant cases
  - Correctly identified sensitive cases
  - Potential unnecessary antibiotics avoided
  - Potential missed resistant cases

- **Feature Importance**
  - Key genetic markers
  - Clinical relevance
  - Potential resistance mechanisms

### 5. Prediction System
- **Robust Prediction Handling**
  - Graceful handling of unknown predictions
  - Confidence score-based decision making
  - Multiple model ensemble predictions
  - Clear display of prediction confidence

- **User Interface Features**
  - Real-time resistance predictions
  - Detailed model-specific results
  - Confidence score visualization
  - Color-coded prediction status
    * Green: Sensitive
    * Red: Resistant
    * Yellow: Unknown/Uncertain

- **Error Handling**
  - Graceful handling of missing models
  - Clear error messages for users
  - Fallback mechanisms for uncertain predictions
  - Detailed logging for debugging

## Setup and Usage

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
cd src
python main.py
```

### Running the Prediction System
```bash
cd src/prediction
python app.py
```

The prediction system will be available at `http://localhost:5000`

## Output and Results

### Visualizations
1. **Distribution Plots**
   - Continent distribution
   - Antibiotic resistance patterns
   - Sensitivity distribution

2. **Geographical Maps**
   - Interactive resistance maps
   - Heatmaps of resistance patterns
   - Combined resistance visualization

3. **Model Performance**
   - ROC curves
   - Confusion matrices
   - Learning curves
   - Feature importance plots

4. **Prediction Results**
   - Summary predictions for each antibiotic
   - Detailed model-specific predictions
   - Confidence scores
   - Best model selection

### Analysis Reports
1. **Continent Analysis**
   - Regional resistance patterns
   - Temporal trends
   - Cross-continent comparisons

2. **Antibiotic Analysis**
   - Resistance patterns
   - Cross-resistance analysis
   - Temporal trends

3. **Clinical Impact**
   - Model performance metrics
   - Potential clinical impact
   - Risk assessment

4. **Prediction System**
   - Model availability status
   - Prediction accuracy metrics
   - Error handling statistics
   - System performance logs

## Future Improvements
1. **Model Enhancements**
   - Deep learning models
   - Ensemble methods
   - Transfer learning

2. **Analysis Extensions**
   - Additional antibiotics
   - More detailed genetic analysis
   - Real-time prediction system

3. **Clinical Integration**
   - Web interface improvements
   - API development
   - Clinical decision support system
   - Enhanced error handling
   - Better handling of edge cases

4. **Prediction System**
   - Real-time model updates
   - Enhanced confidence scoring
   - More detailed prediction explanations
   - Improved handling of unknown cases


