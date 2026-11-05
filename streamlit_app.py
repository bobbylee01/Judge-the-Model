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
    matrix_column, metrics_column = st.columns(2)
    with matrix_column:
        st.caption("TEST confusion matrix · patient counts · rows = Actual, columns = Predicted · positive first")
        st.table(confusion_table(c))
        st.caption("TP: correctly flagged positive patients; FN: positive patients missed; "
                   "FP: negative patients flagged (false alarms); TN: negative patients correctly not flagged.")
    with metrics_column:
        st.caption("TEST key performance metrics · threshold 0.50")
        for metric_row in [
            [("Sensitivity", "sensitivity"), ("Specificity", "specificity")],
            [("Precision", "precision"), ("Accuracy", "accuracy")],
        ]:
            for column, (label, metric) in zip(st.columns(2), metric_row):
                column.metric(f"TEST {label}", m11.fmt_pct(metrics[metric]))
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
    st.write("Explore how the same model's decisions change when you change the probability "
             "needed to flag a patient. Use the sidebar to select a dataset and model.")
    if res["dataset"] == "diabetes":
        st.info("Diabetes screening clinical sensitivity target: ≥ 0.90 (90%). "
                "Of adults meeting the diabetes definition (doctor diagnosis or HbA1c ≥ 6.5%), "
                "the aim is to flag at least 90% for follow-up testing.")
    else:
        st.caption("Positive class: readmitted within 30 days. A positive prediction flags a patient for review.")
    selected = st.slider("Classification threshold", min_value=0.0, max_value=1.0,
                         value=0.5, step=0.01, format="%.2f", key="classification_threshold")
    p, y = m11.preds(res, model, "test")
    fixed_counts = m11.confusion(p, y, 0.5)
    selected_counts = m11.confusion(p, y, selected)
    fixed_metrics = m11.metrics(fixed_counts)
    selected_metrics = m11.metrics(selected_counts)

    st.caption(f"TEST · {MODEL_LABEL[model]} · flag when probability ≥ threshold · "
               "rows = Actual, columns = Predicted · positive first")
    for column, label, value, counts, scores in zip(
        st.columns(2), ["Fixed", "Selected"], [0.5, selected],
        [fixed_counts, selected_counts], [fixed_metrics, selected_metrics],
    ):
        with column:
            st.subheader(f"TEST · {label} threshold {value:.2f}")
            st.caption("TEST confusion matrix · patient counts")
            st.table(confusion_table(counts))
            # The helper's flagged_per_100 uses (TP + FP) / actual TEST n.
            # False alarms per 100 all adults use the same denominator, not FP + TN.
            false_alarms = None if scores["n"] == 0 else 100 * counts["fp"] / scores["n"]
            summary = {
                "Sensitivity (Recall/TPR)": m11.fmt_pct(scores["sensitivity"], 2),
                "Specificity": m11.fmt_pct(scores["specificity"], 2),
                "Precision": m11.fmt_pct(scores["precision"], 2),
                "False positives (false alarms)": f"{counts['fp']:,}",
                "False negatives (missed cases)": f"{counts['fn']:,}",
                "Patients flagged per 100 TEST patients": m11.fmt_num(scores["flagged_per_100"], 2),
                "False alarms per 100 TEST patients": m11.fmt_num(false_alarms, 2),
            }
            st.table(pd.DataFrame(summary.items(), columns=["TEST metric", "TEST value"]).set_index("TEST metric"))
            if res["dataset"] == "diabetes":
                sensitivity = scores["sensitivity"]
                status = ("undefined" if sensitivity is None else
                          "target met" if sensitivity >= 0.90 else "below target")
                st.markdown(f"**TEST sensitivity target: {status}.** "
                            f"At threshold {value:.2f}, there are "
                            f"**{m11.fmt_num(false_alarms, 2)} false alarms per 100 adults** "
                            "in this TEST sample.")

    st.caption("TP: correctly flagged cases; FN: cases missed; FP: false alarms; TN: correctly unflagged non-cases. "
               "Sensitivity: of patients with the outcome, the share flagged. "
               "Specificity: of patients without the outcome, the share not flagged. "
               "Precision: of flagged patients, the share with the outcome. Zero denominators display as undefined.")
    st.markdown(
        f"**TEST change from 0.50:** correctly flagged cases (TP) "
        f"{selected_counts['tp'] - fixed_counts['tp']:+,}; false alarms (FP) "
        f"{selected_counts['fp'] - fixed_counts['fp']:+,}; missed cases (FN) "
        f"{selected_counts['fn'] - fixed_counts['fn']:+,}."
    )
    st.write("Lowering the threshold can detect more cases and miss fewer patients, but can also "
             "flag more patients who do not have the outcome. Counts change only when a patient's "
             "probability crosses the threshold, so some slider steps leave the counts unchanged.")
    st.caption("False alarms per 100 = 100 × FP ÷ all TEST patients. "
               "This differs from FPR, whose denominator is only patients without the outcome. "
               "Patients flagged per 100 = 100 × (TP + FP) ÷ all TEST patients.")
    if res["dataset"] == "diabetes":
        st.write("Lower the slider until the TEST sensitivity target shows 'target met', then read "
                 "the selected threshold's false alarms per 100 adults. Different thresholds may "
                 "meet the target with different follow-up workloads; the choice remains yours.")
    st.metric("TEST AUROC · unchanged by threshold", m11.fmt_num(m11.auroc(p, y)))
    st.write("Changing the threshold changes the decision rule applied to saved probabilities. "
             "It does not retrain the model or change those probabilities. Retraining would fit "
             "the model again on training data. AUROC measures ranking across thresholds, so "
             "moving this slider does not change it; AUROC is not accuracy and does not choose a threshold.")
    st.warning("Choosing a threshold based on TEST performance can make expected performance on "
               "future patients optimistic. This comparison is exploratory; threshold selection "
               "should use separate validation data, with final evaluation on untouched patients.")


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, model: str) -> None:
    st.write("Compare how well each model ranks patients it learned from with patients "
             "it never saw during fitting. Use the sidebar to change dataset or inspect a model.")
    training_count, test_count = st.columns(2)
    training_count.metric("TRAINING patients", f"{res['train']['n']:,}")
    test_count.metric("TEST patients", f"{res['test']['n']:,}")
    s = m11.auroc_summary(res)
    selected = s[model]
    st.subheader(f"TRAINING and TEST · {MODEL_LABEL[model]}")
    for column, label, value in zip(
        st.columns(3),
        ["TRAINING AUROC", "TEST AUROC", "Generalization gap · TRAINING − TEST"],
        [selected["train"], selected["test"], selected["gap"]],
    ):
        column.metric(label, m11.fmt_num(value, 4))

    st.subheader("TRAINING and TEST · All five models")
    rows = [{"Model": MODEL_LABEL[name],
             "TRAINING AUROC": m11.fmt_num(s[name]["train"], 4),
             "TEST AUROC": m11.fmt_num(s[name]["test"], 4),
             "Generalization gap (TRAINING − TEST)": m11.fmt_num(s[name]["gap"], 4)}
            for name in m11.MODELS if name in s]
    st.caption("TRAINING and TEST comparison · larger AUROC means better ranking · "
               "gaps use unrounded AUROCs and are shown in AUROC units")
    st.table(pd.DataFrame(rows).set_index("Model"))

    # Determine leaders from full helper values, preserving ties rather than
    # ranking rounded display values or recommending a model for clinical use.
    for part, label in [("train", "TRAINING"), ("test", "TEST")]:
        available = {name: values[part] for name, values in s.items()
                     if values[part] is not None}
        if available:
            highest = max(available.values())
            leaders = ", ".join(MODEL_LABEL[name] for name, value in available.items()
                                if value == highest)
            st.markdown(f"**Highest {label} AUROC:** {leaders} "
                        f"({m11.fmt_num(highest, 4)}).")
        else:
            st.caption(f"Highest {label} AUROC: undefined.")

    st.markdown(
        "- **TRAINING AUROC** describes ranking on the same patients used to fit the model. "
        "These results flatter the model because it has already learned from these patients.\n"
        "- **TEST AUROC** describes ranking on patients not used to fit the model. "
        "It provides evidence about performance on unseen patients, with uncertainty about future populations.\n"
        "- **Generalization gap (TRAINING − TEST)** shows how much ranking performance falls "
        "from training patients to test patients. A large positive gap suggests overfitting: "
        "the model may have learned details of the training patients that do not carry over. "
        "A negative gap means TEST AUROC exceeded TRAINING AUROC in this sample; "
        "sampling differences can affect either score."
    )
    st.write("A model can rank familiar patients very well yet perform less well on new patients. "
             "Compare the TEST column before judging how well a high TRAINING score carries over; "
             "neither a high TRAINING score nor a small gap guarantees clinical usefulness.")
    st.caption("AUROC describes how often a patient with the outcome receives a higher probability "
               "than a patient without it, with ties receiving half credit. It is not accuracy "
               "and does not choose a classification threshold. Undefined AUROCs or gaps are shown as undefined.")


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
