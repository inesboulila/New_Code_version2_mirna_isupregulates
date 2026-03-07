import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="miRNA Prediction Lab",
    page_icon="🧬",
    layout="wide"
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def strip_prefix(name):
    return re.sub(r'^[a-z]{3}-', '', str(name).lower().strip())

def clean(text):
    return str(text).lower().replace(" ", "").strip()

MODEL_INFO = {
    1: {
        "label"      : "Code 1 — Gradient Boosting Baseline",
        "file"       : "model_code_1.pkl",
        "validation" : "80/20 Split",
        "accuracy"   : "68.75%",
        "auc"        : "0.71",
        "note"       : "Baseline model. Uses 80/20 split so results may vary.",
    },
    2: {
        "label"      : "Code 2 — Honest Validation",
        "file"       : "model_code_2.pkl",
        "validation" : "5-Fold CV",
        "accuracy"   : "63.83%",
        "auc"        : "0.70",
        "note"       : "First honest model. Encoding inside the pipeline prevents leakage.",
    },
    3: {
        "label"      : "Code 3 — GridSearch Best (File A)",
        "file"       : "model_code_3.pkl",
        "validation" : "5-Fold CV",
        "accuracy"   : "66.34%",
        "auc"        : "0.70",
        "note"       : "Best model for Human and Mouse data.",
    },
    4: {
        "label"      : "Code 4 — Leakage Benchmark",
        "file"       : "model_code_4.pkl",
        "validation" : "80/20 Split (Leakage)",
        "accuracy"   : "85.94%",
        "auc"        : "0.95",
        "note"       : "Diagnostic only. Results are inflated and not scientifically valid.",
    },
    5: {
        "label"      : "Code 5 — Scenario Merge",
        "file"       : "model_code_5.pkl",
        "validation" : "5-Fold CV",
        "accuracy"   : "65.11%",
        "auc"        : "0.70",
        "note"       : "Parasite and cell type merged into one scenario feature.",
    },
    6: {
        "label"      : "Code 6 — Blinded Model",
        "file"       : "model_code_6.pkl",
        "validation" : "5-Fold CV",
        "accuracy"   : "65.12%",
        "auc"        : "0.70",
        "note"       : "Species prefixes stripped from miRNA names.",
    },
    10: {
        "label"      : "Code 10 — Dog Model (Recommended)",
        "file"       : "model_code_10.pkl",
        "validation" : "5-Fold CV",
        "accuracy"   : "68.53%",
        "auc"        : "0.75",
        "note"       : "Best overall model. Trained on Human, Mouse and Dog data.",
    },
}

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("🧬 miRNA Prediction Lab")
st.sidebar.markdown("---")

code_options = {v["label"]: k for k, v in MODEL_INFO.items()}
selected_label = st.sidebar.selectbox(
    "Select Model",
    list(code_options.keys()),
    index=list(code_options.keys()).index("Code 10 — Dog Model (Recommended)")
)
v_num = code_options[selected_label]
info  = MODEL_INFO[v_num]

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Details**")
st.sidebar.markdown(f"Validation: `{info['validation']}`")
st.sidebar.markdown(f"Accuracy: `{info['accuracy']}`")
st.sidebar.markdown(f"AUC: `{info['auc']}`")
st.sidebar.info(info["note"])

# ─────────────────────────────────────────────
#  MAIN TITLE
# ─────────────────────────────────────────────

st.title("🧬 miRNA Regulation Predictor")
st.markdown(f"Running: **{info['label']}**")
st.markdown("---")

# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    mirna_raw  = st.text_input("miRNA Name", "hsa-mir-21-5p")
    organism   = st.selectbox("Organism", ["Human", "Mouse", "Dog"])
    parasite   = st.selectbox("Parasite", [
                     "L. donovani", "L. major",
                     "L. infantum", "L. amazonensis"])

with col2:
    cell_type  = st.selectbox("Cell Type", [
                     "PBMC", "THP-1", "BMDM", "RAW 264.7", "HMDM"])
    time_hours = st.number_input("Time (Hours)", min_value=1, value=12)
    if v_num == 10:
        inf_display = st.selectbox("Infection Type",
                                   ["In Vitro", "Naturally Infected"])
    else:
        inf_display = "In Vitro"

predict_btn = st.button("Predict", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────

if predict_btn:
    try:
        loaded = joblib.load(info["file"])

        # Normalize inputs
        p_raw   = mirna_raw.lower().strip()
        p_blind = strip_prefix(p_raw)
        para    = clean(parasite)
        cell    = clean(cell_type)
        org     = organism.lower()
        org_num = 1 if org == "human" else 0
        inf     = "naturallyinfected" if inf_display == "Naturally Infected" else "invitro"

        # ── Build input DataFrame per model ──────────────────────────────

        if v_num == 10:
            # Pipeline: encoder + classifier
            input_df = pd.DataFrame({
                'microrna'                  : [p_blind],
                'microrna_group_simplified' : [p_blind],
                'super_scenario'            : [f"{para}_{cell}_{org}"],
                'infection'                 : [inf],
                'time'                      : [float(time_hours)]
            })
            prediction  = loaded.predict(input_df)[0]
            probability = loaded.predict_proba(input_df)[0][1]

        elif v_num == 6:
            # Pipeline: encoder + classifier, blinded names
            input_df = pd.DataFrame({
                'microrna'                  : [p_blind],
                'microrna_group_simplified' : [p_blind],
                'scenario'                  : [f"{para}_{cell}"],
                'organism'                  : [float(org_num)],
                'time'                      : [float(time_hours)]
            })
            prediction  = loaded.predict(input_df)[0]
            probability = loaded.predict_proba(input_df)[0][1]

        elif v_num == 5:
            # Pipeline: encoder + classifier, scenario feature
            input_df = pd.DataFrame({
                'microrna'                  : [p_raw],
                'microrna_group_simplified' : [p_raw],
                'scenario'                  : [f"{para}_{cell}"],
                'organism'                  : [float(org_num)],
                'time'                      : [float(time_hours)]
            })
            prediction  = loaded.predict(input_df)[0]
            probability = loaded.predict_proba(input_df)[0][1]

        elif v_num == 4:
            # Dict: encoder + xgboost model
            enc = loaded['encoder']
            mdl = loaded['model']
            cat_df = pd.DataFrame({
                'microrna'                  : [p_raw],
                'microrna_group_simplified' : [p_raw],
                'scenario'                  : [f"{para}_{cell}"]
            })
            X_enc = enc.transform(cat_df)
            X_enc['organism'] = float(org_num)
            X_enc['time']     = float(time_hours)
            prediction  = mdl.predict(X_enc)[0]
            probability = mdl.predict_proba(X_enc)[0][1]

        elif v_num == 3:
            # Pipeline from GridSearch
            input_df = pd.DataFrame({
                'microrna'                  : [p_raw],
                'microrna_group_simplified' : [p_raw],
                'parasite'                  : [para],
                'cell type'                 : [cell],
                'organism'                  : [float(org_num)],
                'time'                      : [float(time_hours)]
            })
            prediction  = loaded.predict(input_df)[0]
            probability = loaded.predict_proba(input_df)[0][1]

        elif v_num == 2:
            # Dict: te encoder, ohe encoder, model
            te  = loaded['te']
            ohe = loaded['ohe']
            mdl = loaded['model']

            row = pd.DataFrame({
                'microrna'                  : [p_raw],
                'microrna_group_simplified' : [p_raw],
                'parasite'                  : [para],
                'cell type'                 : [cell],
                'organism'                  : [float(org_num)],
                'time'                      : [float(time_hours)]
            })
            row[['microrna', 'microrna_group_simplified']] = te.transform(
                row[['microrna', 'microrna_group_simplified']])

            ohe_cols = list(ohe.get_feature_names_out(['parasite', 'cell type']))
            ohe_arr  = pd.DataFrame(
                ohe.transform(row[['parasite', 'cell type']]),
                columns=ohe_cols)
            row = row.drop(columns=['parasite', 'cell type'])
            row = pd.concat([row.reset_index(drop=True),
                             ohe_arr.reset_index(drop=True)], axis=1)

            # Align to training columns
            for col in mdl.feature_names_in_:
                if col not in row.columns:
                    row[col] = 0
            row = row[mdl.feature_names_in_]

            prediction  = mdl.predict(row)[0]
            probability = mdl.predict_proba(row)[0][1]

        elif v_num == 1:
            # Dict: target encoder + GB model
            enc = loaded['encoder']
            mdl = loaded['model']

            trained_cols = list(mdl.feature_names_in_)
            input_row    = {c: [0] for c in trained_cols}

            mirna_enc = enc.transform(pd.DataFrame({
                'microrna'                  : [p_raw],
                'microrna_group_simplified' : [p_raw]
            }))
            input_row['microrna']                  = [mirna_enc['microrna'].iloc[0]]
            input_row['microrna_group_simplified']  = [mirna_enc['microrna_group_simplified'].iloc[0]]

            if 'organism' in input_row:
                input_row['organism'] = [float(org_num)]
            if 'time' in input_row:
                input_row['time']     = [float(time_hours)]

            para_col = f"parasite_{para}"
            if para_col in input_row:
                input_row[para_col] = [1]

            cell_col = f"cell type_{cell}"
            if cell_col in input_row:
                input_row[cell_col] = [1]

            X_enc      = pd.DataFrame(input_row)[trained_cols]
            prediction  = mdl.predict(X_enc)[0]
            probability = mdl.predict_proba(X_enc)[0][1]

        # ── Display Results ───────────────────────────────────────────────

        st.markdown("---")
        res1, res2 = st.columns(2)

        with res1:
            if prediction == 1:
                st.success("### UPREGULATED")
            else:
                st.error("### DOWNREGULATED")

        with res2:
            st.metric("Confidence", f"{probability * 100:.1f}%")
            st.progress(float(probability))

        # Confidence note for honest models
        if v_num in [2, 3, 5, 6, 10]:
            st.caption(
                "Note: lower confidence values are expected in honest models "
                "on small datasets. A score of 35% upregulated is still "
                "meaningfully higher than 15% and can be used to rank candidates."
            )

    except FileNotFoundError:
        st.error(f"Model file not found: {info['file']}. Make sure you have run the corresponding notebook cell.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        with st.expander("Technical details"):
            import traceback
            st.code(traceback.format_exc())
