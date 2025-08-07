import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import re
import string
import nltk
import shap
import google.generativeai as genai
import warnings

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

model = tf.keras.models.load_model("task_assign_model.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("onehot_encoder.pkl")

label_map = {
    0: "Anousha Shakeel",
    1: "Fatima",
    2: "Hira",
    3: "Maleeha Asghar",
    4: "Muqaddas Ifikhar",
    5: "Sara"
}

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens) if tokens else "unknown"

def safe_encode_categorical(df, encoder):
    known_columns = encoder.feature_names_in_
    for col in df.columns:
        if col not in known_columns:
            df[col] = "Unknown"
    df = df[known_columns]
    try:
        encoded = encoder.transform(df)
    except ValueError:
        st.error("  One or more values are unknown to the model. Please use valid options.")
        st.stop()
    return encoded

def build_llm_prompt(predicted_assignee, shap_df, top_n=8, task_profile=None):
    intro = f"The model has predicted that this task should be assigned to {predicted_assignee} based on learned patterns from historical task data.\n"

    if task_profile is not None and len(task_profile) > 0:
        intro += (
            f"\nHistorically, {predicted_assignee} has handled tasks involving: "
            f"{', '.join(task_profile)}. The current task description includes similar terms, "
            f"indicating alignment with their past responsibilities.\n"
        )

    intro += "\nThe following SHAP-based feature contributions provide insight into the model's decision:\n\n"
    lines = []

    for _, row in shap_df.head(top_n).iterrows():
        feature = row["Feature"]
        val = row["Input Value"]
        shap_val = row["SHAP Value"]
        direction = "increased" if shap_val > 0 else "decreased"
        impact = abs(shap_val)

        lines.append(f"- Feature '{feature}' with value '{val}' {direction} the model's confidence (impact score: {impact:.3f})")

    reasoning_instructions = """
Generate a concise explanation for the model's task assignment decision in bullet points.

Requirements:
- Each bullet should be a single sentence explaining one key reason for the prediction.
- Summarize the influence of important features (e.g., task description, priority, reporter) without listing raw SHAP values.
- Clearly state how the selected person’s past experience aligns with the current task.
- Briefly mention any weak or conflicting signals and how the model handled them.
- Keep the tone professional and easy to understand for technical and non-technical audiences.

Respond only with bullet points, no paragraphs or summaries.
"""
    return intro + "\n".join(lines) + "\n" + reasoning_instructions

def predict_and_explain_llm(final_input, shap_values, tfidf, encoder, label_map, predicted_class, task_profile, gemini_api_key):
    tfidf_feature_names = tfidf.get_feature_names_out()
    cat_feature_names = encoder.get_feature_names_out()
    all_feature_names = list(tfidf_feature_names) + list(cat_feature_names)

    shap_vals = shap_values[0]
    input_vector = final_input[0]

    shap_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Input Value": input_vector,
        "SHAP Value": shap_vals
    }).sort_values(by="SHAP Value", key=np.abs, ascending=False)

    prompt = build_llm_prompt(label_map[predicted_class], shap_df, task_profile=task_profile)

    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text

# --- Streamlit UI ---
st.set_page_config(page_title="Auto Task Assignment", layout="wide")
st.title( "  Auto Task Assignment Prediction")

with st.form("task_form"):
    description = st.text_area("Task Description", height=150)
    issue_type = st.selectbox("Issue Type", ["Bug", "Incident", "Service Request"])
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
    status = st.selectbox("Status", ["Open", "In Progress", "Resolved", "Closed"])
    creator = st.selectbox("Creator", list(label_map.values()))
    reporter = st.selectbox("Reporter", list(label_map.values()))
    submitted = st.form_submit_button("Predict Assignee")

if submitted:
    if not description.strip():
        st.error("  Please enter a task description.")
        st.stop()

    cat_input_df = pd.DataFrame([[issue_type, priority, reporter, creator, status]],
                                columns=encoder.feature_names_in_)
    encoded_cat = safe_encode_categorical(cat_input_df, encoder)

    clean_text = preprocess_text(description)
    try:
        tfidf_text = tfidf.transform([clean_text]).toarray()
    except NotFittedError:
        st.error("❌ TF-IDF vectorizer is not fitted.")
        st.stop()

    final_input = np.hstack([tfidf_text, encoded_cat])
    prediction = model.predict(final_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    assignee = label_map.get(predicted_class, "Unknown")

    st.success(f"✅ Predicted Assignee: **{assignee}**")

    # 🔍 Get task keywords for context
    nonzero_indices = tfidf_text[0].nonzero()[0]
    top_indices = nonzero_indices[np.argsort(tfidf_text[0][nonzero_indices])[::-1][:5]]
    top_keywords = [tfidf.get_feature_names_out()[i] for i in top_indices]

    #  SHAP explanation
    explainer = shap.KernelExplainer(model.predict, np.random.randn(10, final_input.shape[1]))
    shap_values = explainer.shap_values(final_input)

    #  LLM explanation
    gemini_api_key =  "GEMINI_API_KEY" 
    llm_explanation = predict_and_explain_llm(
        final_input,
        shap_values,
        tfidf,
        encoder,
        label_map,
        predicted_class,
        top_keywords,
        gemini_api_key
    )

    st.subheader("  Explanation:")
    st.markdown(llm_explanation)
