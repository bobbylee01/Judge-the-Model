"""Starter app for Streamlit Community Cloud: three plain sections that read results/ and show correct numbers.

Run locally:  bash run_app.sh  (your agent writes it; it runs streamlit run streamlit_app.py)
Build each section up to its required minimum (see the activity website). Keep every number
coming from m11.py, and label every table and plot "Training" or "Test".
"""

import pandas as pd
import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]
MODEL_LABEL = {**m11.MODEL_LABEL, "knn": "k-nearest neighbors", "lgbm": "LightGBM"}
DATASET_LABEL = {"readmission": "Hospital readmission", "diabetes": "Diabetes screening"}

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def load(name: str):
    return m11.load(name)


def starter_note(text: str) -> None:
    st.info(f"Starter display: {text} Build this section up to its required minimum.")


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis`, then commit and push.")
        for k in m11.DATASETS:
            if any(recs.values()):
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


# ---- 1. Beyond Accuracy ------------------------------------------------------
def confusion_table(c: dict) -> pd.DataFrame:
    """Actual rows and predicted columns, with the positive class first."""
    return pd.DataFrame(
        [[f"TP: {c['tp']:,}", f"FN: {c['fn']:,}"],
         [f"FP: {c['fp']:,}", f"TN: {c['tn']:,}"]],
        index=pd.Index(["Positive", "Negative"], name="TEST Actual"),
        columns=["Predicted positive", "Predicted negative"],
    )


def beyond_accuracy(res: dict, model: str) -> None:
    st.write("Compare correct decisions with missed patients and false alarms. "
             "Accuracy alone can look reassuring when most patients do not have the outcome.")
    is_readmission = res["dataset"] == "readmission"
    positive = ("readmitted within 30 days" if is_readmission else
                "meeting the diabetes definition: doctor diagnosis or HbA1c ≥ 6.5%")
    st.markdown(f"**Positive class:** patients {positive}. "
                "Negative means the patient does not meet this outcome definition.")
    st.caption("TEST patients only · Fixed threshold 0.50: flag a patient when their "
               "predicted probability is at least 0.50. Use the sidebar to change dataset or model.")
    p, y = m11.preds(res, model, "test")
    c = m11.confusion(p, y, 0.5)
    metrics = m11.metrics(c)
    prevalence, majority = st.columns(2)
    prevalence.metric("TEST positive-class prevalence", m11.fmt_pct(metrics["prevalence"]))
    majority.metric("TEST majority-class baseline accuracy", m11.fmt_pct(m11.majority_accuracy(y)))
    st.caption("Prevalence is the share of TEST patients with the outcome. "
               "Majority-class baseline accuracy is the share correct if every TEST patient "
               "were assigned the most common TEST class. The prevalence model below "
               "uses its exported probabilities learned from training patients.")

    st.subheader(f"TEST · {MODEL_LABEL[model]} · threshold 0.50")
    for column, (label, metric) in zip(st.columns(4), [
        ("Accuracy", "accuracy"), ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"), ("Precision", "precision"),
    ]):
        column.metric(f"TEST {label}", m11.fmt_pct(metrics[metric]))
    st.caption("TEST confusion matrix · patient counts · rows = Actual, columns = Predicted · positive first")
    st.table(confusion_table(c))
    st.caption("TP: correctly flagged positive patients; FN: positive patients missed; "
               "FP: negative patients flagged (false alarms); TN: negative patients correctly not flagged.")
    st.markdown(
        f"- **TEST sensitivity (Recall/TPR):** Of patients {positive}, the share flagged "
        "— TP ÷ (TP + FN).\n"
        "- **TEST specificity:** Of patients without the outcome, the share not flagged "
        "— TN ÷ (TN + FP).\n"
        f"- **TEST precision:** Of patients flagged, the share {positive} "
        "— TP ÷ (TP + FP)."
    )
    st.caption('A metric is "undefined" when its denominator is zero; it is never replaced with zero.')

    st.subheader("TEST · Compare all five models · threshold 0.50")
    rows = []
    metric_labels = {
        "accuracy": "Accuracy", "sensitivity": "Sensitivity (Recall/TPR)",
        "specificity": "Specificity", "fpr": "False-positive rate (FPR)",
        "precision": "Precision", "f1": "F1 score",
    }
    # Two matrices per row keep the patient counts and axis labels readable.
    for start in range(0, len(m11.MODELS), 2):
        for column, name in zip(st.columns(2), m11.MODELS[start:start + 2]):
            with column:
                st.markdown(f"**TEST · {MODEL_LABEL[name]} · threshold 0.50**")
                if name not in res["models"]:
                    st.warning("TEST results unavailable for this model.")
                    continue
                model_p, model_y = m11.preds(res, name, "test")
                counts = m11.confusion(model_p, model_y, 0.5)
                scores = m11.metrics(counts)
                st.table(confusion_table(counts))
                rows.append({"TEST Model": MODEL_LABEL[name], **{
                    f"TEST {label}": m11.fmt_pct(scores[metric])
                    for metric, label in metric_labels.items()
                }})
    st.caption("TEST metric comparison · threshold 0.50 · all values are percentages")
    st.table(pd.DataFrame(rows).set_index("TEST Model"))
    st.caption("TEST accuracy: share of all patients correctly classified. "
               "TEST FPR: of patients without the outcome, the share flagged — FP ÷ (FP + TN). "
               "TEST F1: a combined measure of precision and sensitivity — 2TP ÷ (2TP + FP + FN). "
               "Consider missed patients and false alarms alongside accuracy.")


# ---- 2. A Threshold for the Job ------------------------------------------------
def threshold(res: dict, model: str) -> None:
    starter_note("metrics at 0.50 only.")
    p, y = m11.preds(res, model)
    m = m11.metrics(m11.confusion(p, y, 0.5))
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{"Sensitivity": m11.fmt_pct(m["sensitivity"]), "Specificity": m11.fmt_pct(m["specificity"]),
                            "Precision": m11.fmt_pct(m["precision"]), "Flagged per 100": f"{m['flagged_per_100']:.1f}"}]))


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, model: str) -> None:
    starter_note("training and test AUROC of every model, in a table.")
    s = m11.auroc_summary(res)
    rows = [{"Model": m11.MODEL_LABEL[m], f"{TRAIN} AUROC": m11.fmt_num(v["train"]), f"{TEST} AUROC": m11.fmt_num(v["test"])}
            for m, v in s.items()]
    st.caption(f"{TRAIN} and {TEST} · AUROC on the rows each model learned from, and on rows it never saw")
    st.table(pd.DataFrame(rows).set_index("Model"))


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=DATASET_LABEL.get)
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
if model in res["models"]:
    RENDER[section](res, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
