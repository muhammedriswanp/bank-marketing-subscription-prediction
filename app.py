import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# ── Load data & results ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df          = pd.read_csv('data/bank-additional-full.csv', sep=';')
    eval_df     = pd.read_csv('outputs/final_evaluation.csv')
    feat_df     = pd.read_csv('outputs/feature_importance.csv')
    return df, eval_df, feat_df

df, eval_df, feat_df = load_data()

# ── Navigation ─────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Bank Marketing")
st.sidebar.markdown("Predict term deposit subscription")
page = st.sidebar.radio("Navigate", [
    "📊 Model Comparison",
    "📈 Feature Importance",
    "🔍 Predict",
    "🔬 EDA Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Performance comparison of all trained models.")

    # ── Metrics table ──
    st.subheader("All Models — Evaluation Metrics")
    st.dataframe(
        eval_df.style.highlight_max(
            subset=['F1', 'ROC-AUC', 'Recall'],
            color='green'
        ).highlight_min(
            subset=['F1', 'ROC-AUC', 'Recall'],
            color='red'
        ),
        use_container_width=True
    )

    # ── F1 bar chart ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#2ecc71' if '(Tuned)' in m else '#3498db'
                  if 'Baseline' in m else '#95a5a6'
                  for m in eval_df['Model']]
        bars = ax.barh(eval_df['Model'], eval_df['F1'],
                       color=colors, edgecolor='white')
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score by Model')
        ax.axvline(x=eval_df[eval_df['Model'].str.contains('Baseline')]['F1'].values[0],
                   color='red', linestyle='--', alpha=0.5, label='Baseline')
        for bar, val in zip(bars, eval_df['F1']):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("ROC-AUC Comparison")
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(eval_df['Model'], eval_df['ROC-AUC'],
                       color=colors, edgecolor='white')
        ax.set_xlabel('ROC-AUC Score')
        ax.set_title('ROC-AUC by Model')
        for bar, val in zip(bars, eval_df['ROC-AUC']):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Precision vs Recall ──
    st.subheader("Precision vs Recall Trade-off")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(eval_df))
    width = 0.35
    ax.bar(x - width/2, eval_df['Precision'], width,
           label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, eval_df['Recall'], width,
           label='Recall', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_df['Model'], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall per Model')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Best model summary ──
    st.subheader("🏆 Best Model")
    best = eval_df.loc[eval_df['F1'].idxmax()]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model", best['Model'].replace(' (Tuned)', ''))
    col2.metric("F1 Score", best['F1'])
    col3.metric("ROC-AUC", best['ROC-AUC'])
    col4.metric("Precision", best['Precision'])
    col5.metric("Recall", best['Recall'])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    st.markdown("Top features from **Random Forest (Tuned)** model.")

    top_n = st.slider("Show top N features", 5, 25, 15)
    top_df = feat_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.45))
    sns.barplot(
        data=top_df, x='Importance', y='Feature',
        hue='Feature', palette='viridis', legend=False, ax=ax
    )
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest (Tuned)')
    ax.set_xlabel('Importance Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top 5 Feature Interpretations")
    interpretations = {
        'euribor3m'      : "3-month Euro Interbank Offered Rate (3-month maturity) — higher rates make term deposits more attractive",
        'age'            : "Client age — older clients more likely to invest in stable deposits",
        'nr.employed'    : "Number employed — economic stability drives investment confidence",
        'emp.var.rate'   : "Employment variation — job market trends affect financial decisions",
        'campaign'       : "Contact frequency — too many contacts reduce subscription likelihood"
    }
    for feat, desc in interpretations.items():
        st.markdown(f"**{feat}** — {desc}")

    st.subheader("Full Feature Importance Table")
    st.dataframe(feat_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.title("🔍 Predict Term Deposit Subscription")
    st.markdown("Enter customer details to predict subscription likelihood.")

    # ── API status ──
    try:
        r = requests.get("http://127.0.0.1:5000/health", timeout=2)
        if r.status_code == 200:
            st.success("✅ Flask API is running")
        else:
            st.error("❌ Flask API returned error")
    except:
        st.error("❌ Flask API not running — start it with: python flask_api.py")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Client Info")
        age         = st.number_input("Age", 18, 100, 35)
        job         = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur',
                                           'housemaid', 'management', 'retired',
                                           'self-employed', 'services', 'student',
                                           'technician', 'unemployed'])
        marital     = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        education   = st.selectbox("Education", ['university.degree', 'high.school',
                                                  'professional.course', 'basic.9y',
                                                  'basic.6y', 'basic.4y', 'illiterate'])
        housing     = st.selectbox("Housing Loan", ['yes', 'no'])
        loan        = st.selectbox("Personal Loan", ['yes', 'no'])

    with col2:
        st.subheader("📞 Campaign Info")
        contact     = st.selectbox("Contact Type", ['cellular', 'telephone'])
        month       = st.selectbox("Month of Last Contact", ['jan', 'feb', 'mar', 'apr', 'may',
                                             'jun', 'jul', 'aug', 'sep', 'oct',
                                             'nov', 'dec'])
        day_of_week = st.selectbox("Day of Last Contact", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign    = st.number_input("Number of Calls This Campaign", 1, 50, 2)
        previous    = st.number_input("Contacts in Previous Campaigns", 0, 20, 0)
        poutcome    = st.selectbox("Previous Campaign Result", ['nonexistent', 'failure', 'success'])
        previous_contact = 1 if previous > 0 else 0

    with col3:
        st.subheader("📉 Economic Indicators")
        emp_var_rate   = st.number_input("Employment Variation Rate", -5.0, 5.0, -1.8, help="Positive = job growth, Negative = job losses")
        cons_price_idx = st.number_input("Consumer Price Index (Inflation Indicator)", 90.0, 95.0, 92.9, help="Measures inflation — higher means more expensive goods")
        cons_conf_idx  = st.number_input("Consumer Confidence Score", -55.0, 0.0, -46.2, help="How optimistic people feel about the economy")
        euribor3m      = st.number_input("Euribor 3 Month Rate", 0.5, 6.0, 1.3, help="Higher rate = term deposits more attractive")
        nr_employed    = st.number_input("Nr. Employed", 4900.0, 5300.0, 5099.1, help="Total employed persons — higher = stable economy")

    st.divider()

    if st.button("🔍 Predict Subscription", use_container_width=True):
        payload = {
            "age"           : int(age),
            "job"           : job,
            "marital"       : marital,
            "education"     : education,
            "housing"       : housing,
            "loan"          : loan,
            "contact"       : contact,
            "month"         : month,
            "day_of_week"   : day_of_week,
            "campaign"      : int(campaign),
            "previous"      : int(previous),
            "poutcome"      : poutcome,
            "emp.var.rate"  : float(emp_var_rate),
            "cons.price.idx": float(cons_price_idx),
            "cons.conf.idx" : float(cons_conf_idx),
            "euribor3m"     : float(euribor3m),
            "nr.employed"   : float(nr_employed),
            "previous_contact": int(previous_contact)
        }

        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json=payload, timeout=5
            )
            result = response.json()

            if response.status_code == 200:
                col1, col2 = st.columns(2)

                with col1:
                    if result['prediction'] == 1:
                        st.success(f"### ✅ {result['outcome']}")
                    else:
                        st.error(f"### ❌ {result['outcome']}")

                with col2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.bar(['Will Not Subscribe', 'Will Subscribe'],
                           [result['probability_no'], result['probability_yes']],
                           color=['#e74c3c', '#2ecc71'])
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Confidence')
                    for i, v in enumerate([result['probability_no'],
                                           result['probability_yes']]):
                        ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)

                st.subheader("📋 Full API Response")
                st.json(result)
            else:
                st.error(f"API Error: {result}")

        except Exception as e:
            st.error(f"Could not connect to Flask API: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 EDA Insights":
    st.title("🔬 EDA Insights")
    st.markdown("Key findings from Exploratory Data Analysis.")

    col1, col2 = st.columns(2)

    # ── Class distribution ──
    with col1:
        st.subheader("Target Class Distribution")
        counts = df['y'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(counts.index, counts.values,
               color=['#e74c3c', '#2ecc71'], edgecolor='white')
        ax.set_xlabel('Subscribed')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution (Imbalanced)')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)',
                    ha='center', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Subscription by poutcome ──
    with col2:
        st.subheader("Subscription Rate by Previous Outcome")
        pout = df.groupby('poutcome')['y'].apply(
            lambda x: (x == 'yes').sum() / len(x) * 100
        ).reset_index()
        pout.columns = ['poutcome', 'subscription_rate']
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=pout, x='poutcome', y='subscription_rate',
                    hue='poutcome', palette='Set2', legend=False, ax=ax)
        ax.set_ylabel('Subscription Rate (%)')
        ax.set_title('Subscription Rate by Previous Outcome')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Correlation heatmap ──
    st.subheader("Correlation Heatmap — Numerical Features")
    num_cols = ['age', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax, linewidths=0.5)
    ax.set_title('Correlation Matrix — Numerical Features')
    plt.tight_layout()
    st.pyplot(fig)

    # ── Key findings ──
    st.subheader("📋 Key EDA Findings")
    findings = [
        "**Class Imbalance** — 88.7% No / 11.3% Yes → used F1 & ROC-AUC as metrics",
        "**poutcome = success** → strongest predictor, ~65% subscription rate",
        "**euribor3m, emp.var.rate, nr.employed** → highly correlated economic indicators",
        "**Cellular contact** → higher subscription rate than telephone",
        "**duration dropped** → data leakage (unknown before call ends)",
        "**housing & loan unknowns** → same 796 rows, imputed with mode",
    ]
    for f in findings:
        st.markdown(f"- {f}")