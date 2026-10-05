# Student Skill Analyzer

A beginner-to-intermediate Python project that analyzes fictional student skill scores. It uses Streamlit for the interface, Pandas and NumPy for data work, Plotly for charts, and a small scikit-learn decision tree as an introduction to machine learning.

## Features

- Load the included sample CSV or upload another CSV.
- Compare average scores across Python, Mathematics, Statistics, and Communication.
- View individual student profiles and a results table.
- Suggest a practice area from the lowest score.
- Train a small decision tree on the sample `Learning_Focus` labels and try it with entered scores.

## Run it

You need Python 3.10 or newer. Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the libraries and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`) to open in your browser. In IBM Bob, open this project folder, ask Bob to explain or modify `app.py`, and use the same terminal commands to run the app.

## Use your own CSV

The uploaded CSV must contain these columns (spelling and capitalization matter):

```text
Student,Python,Mathematics,Statistics,Communication
```

Scores should be numeric values from 0 to 100. Rows with missing or non-numeric skill scores are skipped. Scores below or above the range are clipped to 0 or 100. `Learning_Focus` is optional: without labeled rows, charts and analysis still work, but the decision-tree demo is unavailable.

## Folder layout

```text
student-skill-analyzer/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── sample_students.csv
```

## Learning notes and limitations

The included names and scores are fictional. In the sample file, `Learning_Focus` is made from the lowest skill score. The model learns this simple label pattern from only 12 rows, so its output is for practicing the ML workflow, not for assessing real students or making high-stakes decisions. A real predictive model would need a larger, representative dataset, meaningful outcomes, and careful evaluation on data the model did not train on.

Suggested next steps: add a subject filter, let the user download analyzed results, then learn train/test splits and model evaluation before trying a more realistic prediction task.
