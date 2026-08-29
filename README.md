# README: Beyond literal meaning: an idiom-aware framework for Indonesian emotion classification using hybrid Natural Language Processing (NLP) models

## Project Title

**Beyond literal meaning: an idiom-aware framework for Indonesian emotion classification using hybrid Natural Language Processing (NLP) models**

## Description

This repository contains the official computer code and structural environment for the hybrid Natural Language Processing (NLP) models designed to classify emotions in Indonesian text by incorporating idiom-aware frameworks. It implements an architecture integrating IndoBERT with K-Nearest Neighbors (IndoBERT-KNN) and handles evaluation metrics including the Area Under the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) analysis (ROC-AUC).

## Dataset Information

- **Data Directory Structure:** All research datasets are securely structured under the `backend/data/` directory.
  - **Raw Data:** Located in `backend/data/raw/` (Contains the original data files formatted in English).
  - **Preprocessed Data:** Located in `backend/data/preprocessed/` (Contains engineered features for model consumption).
  - **Testing Data:** Located in `backend/data/testing/` (Contains validation splits such as holdout subsets).
- **Data Language:** English (Completely translated and aligned with PeerJ requirements for reviewer and reader accessibility). The primary reference file uploaded to the submission portal is designated as `Dataset.csv`.

## Prerequisites & Requirements

- **Python Runtime:** Version 3.14.5 (or compatible 64-bit AMD architecture).
- **Database Engine:** PostgreSQL Version 18.3.2.
- **IDE/Editor:** Visual Studio Code (VSCode) recommended.
- **Core Dependencies:** Python libraries listed in `requirements.txt`.

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
4. **Validation:** Evaluates predictions using multi-class Area Under the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) analysis.

## Code Availability & Open Data Policy

In compliance with the PeerJ open data policy, this code is formatted for review and publication validation by the designated Academic Editor and reviewers.

## License & Contribution Guidelines

This project is submitted exclusively under peer-review restrictive terms for PeerJ evaluation. All rights reserved.
