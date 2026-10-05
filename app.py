"""A beginner-friendly dashboard for exploring student skills and study focus."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).parent / "data" / "sample_students.csv"
SKILL_COLUMNS = ["Python", "Mathematics", "Statistics", "Communication"]
REQUIRED_COLUMNS = ["Student", *SKILL_COLUMNS]

st.set_page_config(page_title="Student Skill Analyzer", page_icon="🎓", layout="wide")


@st.cache_data
def load_sample_data():
    """Load the small example dataset included with this project."""
    return pd.read_csv(DATA_PATH)


def clean_data(data):
    """Check columns and turn skill scores into usable numbers."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    cleaned = data.copy()
    for column in SKILL_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=SKILL_COLUMNS)
    for column in SKILL_COLUMNS:
        cleaned[column] = cleaned[column].clip(0, 100)
    return cleaned


def recommend_focus(scores):
    """Return the skill with the lowest score as a simple study suggestion."""
    return min(SKILL_COLUMNS, key=lambda skill: scores[skill])


def train_demo_model(data):
    """Train a tiny decision tree on the sample data's rule-based focus labels."""
    if "Learning_Focus" not in data.columns or len(data) < 3:
        return None
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(data[SKILL_COLUMNS], data["Learning_Focus"])
    return model


st.title("🎓 Student Skill Analyzer")
st.write(
    "Explore skill scores, find strengths, and get a starting point for what to practice next. "
    "This is a learning project: its sample data is fictional and its model is only a demonstration."
)

with st.sidebar:
    st.header("Choose your data")
    uploaded_file = st.file_uploader("Upload a CSV (optional)", type="csv")
    st.caption("Required columns: Student, Python, Mathematics, Statistics, Communication")
    st.caption("Scores should be from 0 to 100. Extra columns are allowed.")

try:
    raw_data = pd.read_csv(uploaded_file) if uploaded_file else load_sample_data()
    students = clean_data(raw_data)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"I couldn't read this dataset: {error}")
    st.stop()

if students.empty:
    st.warning("There are no rows with complete numeric skill scores to analyze.")
    st.stop()

# NumPy calculates the mean score for each student; Pandas handles the table.
students["Average"] = np.round(students[SKILL_COLUMNS].to_numpy().mean(axis=1), 1)
students["Rule_Based_Focus"] = students[SKILL_COLUMNS].idxmin(axis=1)

st.subheader("At a glance")
first, second, third = st.columns(3)
first.metric("Students analyzed", len(students))
second.metric("Group average", f"{students['Average'].mean():.1f} / 100")
strongest = students[SKILL_COLUMNS].mean().idxmax()
third.metric("Highest group skill", strongest)

left, right = st.columns(2)
with left:
    st.subheader("Average score by skill")
    averages = students[SKILL_COLUMNS].mean().rename_axis("Skill").reset_index(name="Average score")
    figure = px.bar(
        averages, x="Skill", y="Average score", color="Average score",
        color_continuous_scale="Teal", range_y=[0, 100], text_auto=".1f",
    )
    figure.update_layout(coloraxis_showscale=False, yaxis_title="Score out of 100")
    st.plotly_chart(figure, use_container_width=True)

with right:
    st.subheader("Student skill profiles")
    chosen = st.multiselect(
        "Students to compare", students["Student"].astype(str).tolist(),
        default=students["Student"].astype(str).tolist()[: min(4, len(students))],
    )
    selected = students[students["Student"].astype(str).isin(chosen)]
    if not selected.empty:
        long_data = selected.melt(id_vars="Student", value_vars=SKILL_COLUMNS, var_name="Skill", value_name="Score")
        figure = px.line(long_data, x="Skill", y="Score", color="Student", markers=True, range_y=[0, 100])
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Choose one or more students to show their skill profiles.")

st.subheader("Student-by-student results")
display = students[["Student", *SKILL_COLUMNS, "Average", "Rule_Based_Focus"]].rename(
    columns={"Rule_Based_Focus": "Suggested practice"}
)
st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("Try the demo ML model")
st.write(
    "The sample file includes a `Learning_Focus` label created from each student's lowest score. "
    "A small decision tree learns that pattern and suggests a focus for scores you enter. "
    "This demonstrates a basic ML workflow; it is not a validated prediction about a real student."
)
model = train_demo_model(students)
if model is None:
    st.info("The uploaded data has no Learning_Focus labels. Analysis still works; ML needs labeled sample rows.")
else:
    with st.form("new_student_form"):
        st.write("Enter scores for a new example")
        name = st.text_input("Student name", "New student")
        input_columns = st.columns(4)
        scores = {}
        for column, skill in zip(input_columns, SKILL_COLUMNS):
            scores[skill] = column.slider(skill, 0, 100, 60)
        submitted = st.form_submit_button("Suggest a practice focus")
    if submitted:
        input_frame = pd.DataFrame([scores], columns=SKILL_COLUMNS)
        predicted = model.predict(input_frame)[0]
        simple_rule = recommend_focus(scores)
        st.success(f"{name}: the demo model suggests practicing **{predicted}** first.")
        st.caption(f"Simple lowest-score rule: {simple_rule}. The tree may differ because it learned from a very small sample.")

with st.expander("What to learn from this project"):
    st.markdown(
        "- **Pandas:** reads CSV files, cleans columns, and prepares tables.\n"
        "- **NumPy:** calculates each student's average from the score values.\n"
        "- **Charts:** compare group averages and individual profiles.\n"
        "- **Scikit-learn:** fits a small decision tree and predicts a focus label.\n"
        "- **Streamlit:** turns the Python script into an interactive web app."
    )
