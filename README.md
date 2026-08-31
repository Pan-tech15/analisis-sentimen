# README: Beyond literal meaning: an idiom-aware framework for Indonesian emotion classification using hybrid Natural Language Processing (NLP) models

## Project Title

**Beyond literal meaning: an idiom-aware framework for Indonesian emotion classification using hybrid Natural Language Processing (NLP) models**

## Description

This repository contains the official computer code and structural production environment for the idiom-aware emotion classification frameworks developed for Indonesian text. It implements a highly optimized, modular processing pipeline integrating idiom detection, idiom semantic interpretation, and emotion classification. This platform comprehensively implements and evaluates two distinct hybrid NLP paradigms: an architecture integrating IndoBERT with instance-based learning via K-Nearest Neighbors coupled with Uniform Manifold Approximation and Projection (IndoBERT–KNN) for high-dimensional contextual embeddings, and an ensemble Naïve Bayes–Lexicon approach incorporating statistical Pointwise Mutual Information (PMI) and mathematical Soft-Probability Decision Fusion. The codebase handles exhaustive validation metrics including Area Under the Receiver Operating Characteristic curve and Area Under the Curve analysis (ROC–AUC), Matthews Correlation Coefficient (MCC), Log Loss tracking, and multi-class macro performance averages.

## Dataset Information

- **Data Directory Structure:** All research corpuses are securely managed under the backend storage structure:
  - **Raw Ingestion Directory:** `backend/data/raw/` (Contains original text files formatted with English metadata mapping).
  - **Engineered Feature Directory:** `backend/data/preprocessed/` (Contains pre-computed linguistic tokens and feature matrix arrays).
  - **Validation Holdout Directory:** `backend/data/testing/` (Contains independent validation slices generated at runtime).
- **Data Language:** English metadata structures (The input text entries are in Indonesian, while all classification metadata, schemas, column headers, and target labels are fully mapped in English to ensure absolute reviewer accessibility). The definitive deployed file reference is designated as `Dataset.csv`.

### Data Distribution Summary

The underlying research model pipeline processes a balanced data repository consisting of **10,186 text samples** compiled during the primary ingestion phase. The corpus maintains a natural balance distributed across core target emotion categories and literal text classes, which includes:

- **Non-idiom Baseline Data:** 5,013 samples (Automatically categorized via the presence of the `has_idiom` flag).
- **Idiomatic Context Data Classes:** 5,173 idiomatic sentences assigned into 7 emotion categories based on Plutchik's taxonomy:
  - happiness (767 samples)
  - trust (734 samples)
  - surprise (735 samples)
  - neutral (724 samples)
  - sadness (733 samples)
  - fear (744 samples)
  - anger (736 samples)

Because the macro distribution between the non-idiomatic category and the cumulative idiomatic configurations inherently reflects an optimized pattern (approximate 50:50 macro distribution), the dataset does not require synthetic data generation (Augmentation) or random under-sampling algorithms (Discarding). Consequently, the model pipelines leverage the authentic empirical architecture of the data without external modifications. The complete distribution frequencies are detailed in **Table 1** of the main manuscript.

## Code Information

The production repository architecture follows a modular, layered structure designed to strictly separate database interaction, API routing, core business logic, and utility pipelines:

- `backend/run.py`: The application entry point that instantiates, configures, and boots the local Flask development server.
- `backend/app/__init__.py`: Initializes the Flask application package, configures global extensions, and binds blueprints.
- `backend/app/config.py`: Manages environment configurations, structural application parameters, and PostgreSQL database credentials.
- `backend/app/models/`: Contains database schema definitions and Object-Relational Mapping (ORM) models for tracking model runs (`training.py`).
- `backend/app/routes/`: Houses modular API endpoints and routes divided by functional domains (e.g., `dashboard.py`) to handle requests from the frontend client.
- `backend/app/services/`: Implements the core business logic, containing the main training engines for the machine learning pipelines (`train_indobert_knn` and `train_lexicon_nb`).
- `backend/app/utils/`: Dedicated directory for static evaluation metrics, database helper scripts, and text preprocessing tokenization routines (`preprocessing_utils.py`).
- `backend/migrations/`: Stores automated database migration scripts and schema versioning histories managed via Flask-Migrate.
- `frontend/`: Contains presentation layer files, client-side web assets, and tracking dashboard interfaces.

## Prerequisites & Requirements

- **Python Runtime:** Version 3.12.5 (or compatible 64-bit AMD architecture).
- **Database Engine:** PostgreSQL Version 16.2 (Target database: `analisis_sentimen`).
- **Core Dependencies:** Python libraries listed in `requirements.txt` (including `torch`, `transformers`, `umap-learn`, `joblib`, `scikit-learn`, and `Sastrawi`).

### Model Configuration & Hyperparameters

To ensure exact replication of the empirical results presented in the study, the models must be configured with the specific hyperparameter baselines listed below:

#### IndoBERT Architecture Settings

- **Base Pretrained Model:** `indobenchmark/indobert-base-p2`
- **Training Strategy:** Fixed epoch-based fine-tuning (Epochs = 5) without internal cross-validation for parameter selection. The complete parameter configuration is summarized in **Table 4** of the manuscript.
- **Max Sequence Length:** 128 tokens | **Batch Size:** 32 samples | **Pooling Layer:** MEAN aggregation
- **Fine-Tuning Hyperparameters:** Learning Rate = 2E-05, Optimizer = AdamW, Weight Decay = 0.01, Warmup Ratio = 0.1, Freeze Encoder Layers = 9, Classifier Dropout = 0.1.

#### Dimension Reduction & Classifier (UMAP + KNN)

- **UMAP Components:** n_components = 200, n_neighbors = 30, min_dist = 0.1, metric = 'cosine' (parameter details available in **Table 5**).
- **KNN Classifier:** K = 5, metric = 'cosine', weights = 'distance', algorithm = 'auto', leaf_size = 30.

#### Naïve Bayes–Lexicon Classifier Settings

- **Training Strategy:** Hyperparameters (e.g., Laplace smoothing alpha) are optimized via a Stratified 10-fold Cross-Validation approach strictly within the training subset.
- **Classifier Type:** Multinomial Naïve Bayes (`MultinomialNB`) with Laplace smoothing parameter alpha = 0.3.
- **Feature Vectorizer:** `TfidfVectorizer` used to extract numerical text features by representing term importance within the text corpus.
- **Hybrid Fusion Method:** Weighted Soft-Probability Decision Fusion ($P_{\text{final}} = \alpha \cdot P_{\text{NB}} + (1 - \alpha) \cdot S_{\text{lexicon}}$) with weight $\alpha = 0.9$.
- **Tie-Breaker Ranking Array:** 7-tier scale mapping priority values: Happy (1), Trust (2), Surprised (3), Neutral (4), Sad (5), Scared (6), Angry (7).

## Environment Setup & Installation

### 1. Database Configuration & Migration

1. Install PostgreSQL version 16.2 on your system.
2. Open pgAdmin4 or your preferred SQL terminal, connect to your local server, and create a new empty database named exactly `analisis_sentimen`.
3. Run the database upgrade command inside your activated virtual environment to automatically generate all training tables:
   ```bash
   cd backend
   flask db upgrade
   ```

### 2. Project Directory & Dataset Initialization

1. Navigate to the `backend` project folder.
2. Place your dataset files (provided as supplementary material `Dataset.csv`) within their respective subfolders under `backend/data/` (i.e., raw files inside `/raw`, preprocessed files inside `/preprocessed`, and evaluation splits inside `/testing`).
3. Create a new environment configuration file named `.env` inside the `backend` folder.
4. Populate the `.env` file with the connection string and credentials matching your local PostgreSQL setup. Copy the template from `env.txt` and adjust the `DATABASE_URL` value (username, password, host, port).

### 3. Dependency Installation

1. Open the project root folder inside VSCode terminal and initialize a Python virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
3. Install all required dependencies from the package registry:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions (Running the Application)

### Backend Server Execution

1. Ensure your terminal environment indicates an active environment marker: `(venv)`.
2. Move into the backend subdirectory:
   ```bash
   cd backend
   ```
3. Boot the application engine:
   ```bash
   python run.py
   ```
   _The Flask backend API service will securely bind and expose routing endpoints on `http://localhost:5000`._

### Frontend Web Server Execution

1. Open a new terminal instance within the workspace (keep the backend server running).
2. Move into the frontend subdirectory:
   ```bash
   cd frontend
   ```
3. Spin up the lightweight HTTP hosting server on port 5500:
   ```bash
   python -m http.server 5500
   ```

### Access Ports & Local URLs

Once both servers are running successfully, access the interfaces via:

- **Administrative Portal:** `http://localhost:5500/admin/login.html`
- **Public Dashboard:** `http://localhost:5500/public/beranda.html`

_Default System Administrative Credentials:_

- **Username:** `admin` | **Password:** `admin123`

## Methodology & Architectural Workflow

The execution pipeline follows the 4 structured framework stages established in the study: data collection, pre-processing, processing, and evaluation.

### Stage 1: Data Collection & Stage 2: Pre-Processing

Linguistic text cleaning processes are executed through tailored configurations depending on the requirements of each model framework:

- **Context-Preserving Preprocessing:** Implements text case folding and slang-word mapping to standardize the language while retaining complete structural dependencies. Applied specifically for the **IndoBERT–KNN** framework.
- **Aggressive Token-Level Cleansing:** Implements rigorous linguistic cleaning including lowercase transformation, punctuation removal, stopword filtering utilizing a manual token dictionary, and grammatical word-stemming via Sastrawi. Applied specifically for the **Naïve Bayes–Lexicon** approach.

### Stage 3: Processing Procedure

The comprehensive corpus is split into training and testing sets using a **hold-out strategy**. As detailed in **Table 3**, the optimal split ratio differs between the two models: **IndoBERT–KNN achieves peak performance at an 85:15 split**, whereas **Naïve Bayes–Lexicon performs optimally at an 80:20 split**.

- **IndoBERT-KNN Pipeline (`train_indobert_knn`):**
  Sequentially coordinates `Idiom detection` checks via the `has_idiom` constraint, `Idiom meaning annotation`, and contextual encoder pooling. The core IndoBERT model is fine-tuned using a fixed **5 Epoch** strategy (hyperparameters detailed in **Table 4**) applied directly to the training set. Following encoding, sentence embeddings are projected into a compact semantic space using **UMAP** (parameter configuration in **Table 5**) and classified via KNN.

- **Naïve Bayes-Lexicon Approach (`train_lexicon_nb`):**
  Constructs a dynamic co-occurrence grid comprising dictionary lookup scoring and corpus-wide Pointwise Mutual Information (`compute_pmi`). Unlike the epoch-based training of IndoBERT, **hyperparameter optimization (e.g., the `alpha` smoothing parameter) is conducted internally via a Stratified 10-fold Cross-Validation** strictly within the training subset. Features are vectorized using a TF-IDF scheme, fed into a `MultinomialNB` engine, and integrated via a custom soft-probability mathematical fusion algorithm.

### Stage 4: Evaluation & Model Performance Highlights

Model performance is executed via Confusion Matrices alongside standardized validation configurations:

- **Validation Framework:** While hyperparameter tuning for the Naïve Bayes–Lexicon model utilizes Stratified 10-fold CV on the training data, the **final generalization performance for both models** is rigorously validated on the separate independent testing holdout (using the optimal split ratios specified in Table 3).
- **Core Performance Metrics:** System validation tracks macro-averaged parameters across accuracy, precision, recall, F1-score, Log Loss, Matthews Correlation Coefficient (MCC), and ROC–AUC. The consolidated training and testing results for all metrics are presented in **Table 6** of the manuscript.
- **Experimental Baselines:** When validated on the independent holdout, the proposed **IndoBERT–KNN** framework with UMAP achieved a prominent accuracy and F1-score of **97.45%** (Testing Loss: 0.0265, MCC: 0.970). This performance systematically outpaced the baseline **Naïve Bayes–Lexicon** model configuration, which achieved **93.13%** accuracy under identical testing splits. An ablation setup executed entirely without UMAP dimensionality reduction resulted in a lower testing accuracy of **89.07%** and an F1-score of **88.73%** for the transformer configuration.

## Code Availability & Open Data Policy

In compliance with the PeerJ open data policy, the source code, environmental configurations, and underlying text corpuses are temporarily archived for publication review. Permanent static archives tracking definitive release tags will be accessible via Zenodo DOI upon formal publication.

## License & Contribution Guidelines

This project is submitted exclusively under peer-review restrictive terms for PeerJ evaluation. All rights reserved.
