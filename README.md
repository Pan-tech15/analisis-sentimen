# README: Beyond literal meaning: an idiom-aware framework for Indonesian emotion classification using hybrid Natural Language Processing (NLP) models

## Project Title

**Beyond literal meaning: an idiom-aware framework for Indonesian emotion classification using hybrid Natural Language Processing (NLP) models**

## Description

This repository contains the official computer code and structural environment for the idiom-aware emotion classification frameworks developed for Indonesian text. It implements a unified processing pipeline integrating idiom detection, idiom interpretation, and emotion classification. This platform evaluates two distinct hybrid NLP paradigms: an architecture integrating IndoBERT with K-Nearest Neighbors using Uniform Manifold Approximation and Projection (IndoBERT–KNN) for contextual representations, and a Naïve Bayes–Lexicon approach incorporating emotion intensity-based lexicons. The codebase handles full evaluation metrics including the Area Under the Receiver Operating Characteristic curve and Area Under the Curve analysis (ROC–AUC), Matthews Correlation Coefficient (MCC), and multi-class macro performance averages.

## Dataset Information

- **Data Directory Structure:** All research datasets are securely structured under the `backend/data/` directory.
  - **Raw Data:** Located in `backend/data/raw/` (Contains the original data files formatted in English).
  - **Preprocessed Data:** Located in `backend/data/preprocessed/` (Contains engineered features for model consumption).
  - **Testing Data:** Located in `backend/data/testing/` (Contains validation splits such as holdout subsets).
- **Data Language:** English (Completely translated and aligned with PeerJ requirements for reviewer and reader accessibility). The primary reference file uploaded to the submission portal is designated as `Dataset.csv`.

### Data Distribution Summary

The underlying research utilizes a dataset initially consisting of **10,186 text samples** compiled during the primary ingestion phase. The corpus maintains a natural balance distributed across core target emotion categories and literal text classes, which includes:

- **Non-idiom data:** 5,013 samples
- **Idiomatic data classes:** Distributed across 5,173 idiomatic sentences assigned into 7 emotion categories based on Plutchik's taxonomy:
  - happiness (767)
  - trust (734)
  - surprise (735)
  - neutral (724)
  - sadness (733)
  - fear (744)
  - anger (736)

Because the macro distribution between the non-idiomatic category and the cumulative idiomatic configurations inherently reflects a balanced pattern (approximate 50:50 distribution), the dataset does not require synthetic data generation (Augmentation) or random under-sampling algorithms (Discarding). Consequently, the model pipelines leverage the authentic empirical architecture of the data without external modifications. Clear distribution frequencies across the model frameworks can be fully referenced in **Table 1** of the main manuscript.

## Code Information

The project architecture follows a modular, layered production structure designed to strictly separate database interaction, API routing, core business logic, and utility pipelines:

- `backend/run.py`: The application entry point that instantiates, configures, and runs the Flask development server.
- `backend/app/__init__.py`: Initializes the Flask application package, configures extensions, and binds application blueprints.
- `backend/app/config.py`: Manages global environment configurations, runtime parameters, and PostgreSQL database credentials.
- `backend/app/models/`: Contains database schema definitions and Object-Relational Mapping (ORM) models for storing persistent data (e.g., text, idioms, and classifications).
- `backend/app/routes/`: Houses modular API endpoints and routes divided by functional domains (e.g., `dashboard.py`) to handle requests from the frontend client.
- `backend/app/services/`: Implements the core business logic, data processing workflows, and the IndoBERT embedding and K-Nearest Neighbors (KNN) execution frameworks.
- `backend/app/utils/`: Dedicated directory for helper scripts, text preprocessing routines, and static evaluation metrics.
- `backend/migrations/`: Stores automated database migration scripts and schema versioning histories managed via Flask-Migrate.
- `frontend/`: Contains the presentation layer files, client-side web assets, and interfaces for the Administrative Portal and Public Dashboard.

## Prerequisites & Requirements

- **Python Runtime:** Version 3.14.5 (or compatible 64-bit AMD architecture).
- **Database Engine:** PostgreSQL Version 18.3.2.
- **IDE/Editor:** Visual Studio Code (VSCode) recommended.
- **Core Dependencies:** Python libraries listed in `requirements.txt`.

### Model Configuration & Hyperparameters

To ensure exact replication of the empirical results presented in the study, the models must be configured with the specific hyperparameter baselines listed below:

#### IndoBERT Architecture Settings

- **Max Sequence Length:** 128 | **Batch Size:** 16 | **Pooling Layer:** Mean
- **Epochs:** 5 (Fine-tuning phase) | **Learning Rate:** 2E-05 | **Optimizer:** AdamW
- **Freeze Layers:** 9 | **Classifier Dropout:** 0.1 | **Target Dimension:** 768
  _(For complete layers setup, see **Table 4** in the manuscript)_

#### Dimension Reduction & Classifier (UMAP + KNN)

- **UMAP Components:** 200 components with 30 neighbors (`metric='cosine'`, `min_dist=0.1`)
- **KNN Classifier:** K=5 (`metric='cosine'`, `weights='distance'`, `algorithm='auto'`)
  _(Detailed configurations are mapped in **Table 5** of the manuscript)_

## Environment Setup & Installation

### 1. Database Configuration & Migration

1. Install PostgreSQL version 18.3.2 on your system.
2. Open pgAdmin4 or your preferred SQL terminal, connect to your local server, and create a new empty database named exactly `analisis_sentimen`.
3. Since this project utilizes Flask-Migrate for database schema versioning, run the database upgrade command inside your activated virtual environment to automatically generate all tables (including the `idioms` table):
   ```bash
   cd backend
   flask db upgrade
   ```

### 2. Project Directory & Dataset Initialization

1. Navigate to the `backend` project folder.
2. Ensure your English-translated dataset files are placed within their respective subfolders under `backend/data/` (i.e., placing raw files inside `/raw`, preprocessed files inside `/preprocessed`, and evaluation splits inside `/testing`).
3. Create a new environment configuration file named `.env` inside the `backend` folder.
4. Populate the `.env` file with the connection string and credentials matching your local PostgreSQL setup as outlined in `env.txt`.

### 3. Dependency Installation

1. Open the project root folder (`analisis-sentimen`) inside VSCode.
2. Open the integrated terminal (`Ctrl + ~`) and initialize a Python virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
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
4. Install all required dependencies from the package registry:
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

- **Username:** `admin`
- **Password:** `admin123`

## Methodology & Architectural Workflow

1. **Data Ingestion:** Reads structured text entries from the data directory.
2. **Preprocessing:** Tokenizes and extracts embedding values optimized for Indonesian idiom phrases.
3. **Classification:** Employs a hybrid model fusing IndoBERT architectures with K-Nearest Neighbors (KNN) logic.
4. **Validation:** Evaluates predictions using multi-class Area Under the Receiver Operating Characteristic curve and Area Under the Curve analysis.

### Model Evaluation Highlights

The repository comprehensively evaluates and contrasts two distinct hybrid Natural Language Processing (NLP) paradigms implemented across the data architecture: our primary **IndoBERT–KNN** framework (leveraging contextual language representations with instance-based learning) and the baseline **Naïve Bayes–Lexicon** configuration (integrating probabilistic classification with emotion intensity-based lexicons).

- **Sample Corpus Baseline:** Both algorithmic pipelines process the baseline dataset consisting of **10,186 text samples** distributed across core target emotion and literal text classes.
- **Class Stratification:** This includes 5,013 non-idiomatic references and an even macro spread ranging from 724 to 767 samples per individual idiomatic emotion category.
- **Core Framework Results:** On the testing evaluation partitions, the proposed **IndoBERT–KNN** framework achieved a prominent accuracy and F1-score of **97.45%**, outperforming the baseline **Naïve Bayes–Lexicon** model, which achieved **93.13%**.
- **Ablation Study Parameters:** Notably, a comparative experimental evaluation executed entirely without UMAP dimensionality reduction resulted in a lower testing accuracy of **89.07%** and an F1-score of **88.73%** for the baseline IndoBERT configuration.
- **Manuscript Mapping:** Exact distribution frequencies and multi-metric performance breakdowns across stratified K-fold cross-validation setups can be referenced in **Table 1** and **Table 6** of the main manuscript.

## Code Availability & Open Data Policy

In compliance with the PeerJ open data policy, the source code, environmental configurations, and underlying text corpuses are temporarily archived for publication review. Permanent static archives tracking definitive release tags will be accessible via Zenodo DOI upon formal publication.

## License & Contribution Guidelines

This project is submitted exclusively under peer-review restrictive terms for PeerJ evaluation. All rights reserved.
