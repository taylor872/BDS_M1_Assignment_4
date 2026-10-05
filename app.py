import numpy as np
import pandas as pd
import streamlit as st

import portable  # the helper you downloaded in step 9

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
FEATURES = ["age", "gender", "programme", "evening_programme", "admission_grade",
            "international", "first_gen", "su_scholarship", "fees_owed", "moved_from_home",
            "married", "logins_total", "logins_last3", "logins_trend", "submitted_share",
            "missed_last3", "quiz_mean", "weeks_since_login"]

st.set_page_config(page_title="Study office · week 6", layout="wide")


@st.cache_resource
def load_model():
    return portable.Model("model")          # reads model/booster.json + model/preprocess.json


@st.cache_data
def load_data():
    model = load_model()
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    val = history[history["cohort"] == 2025].copy()        # last year's students: we know who left
    val["risk"] = model.predict_proba(val[FEATURES])
    new["risk"] = model.predict_proba(new[FEATURES])
    return val, new


val, new = load_data()


def flagged_by(df, mode, value):
    """Which students does the rule contact?"""
    if mode == "Number of conversations":
        return df["risk"].rank(ascending=False, method="first") <= value
    return df["risk"] >= value


def boxes(df, flagged):
    left = df["left"] == 1
    tp = int((flagged & left).sum())
    fp = int((flagged & ~left).sum())
    fn = int((~flagged & left).sum())
    tn = int((~flagged & ~left).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, tn, precision, recall


# ---- sidebar: the rule ----
st.sidebar.header("The rule")
mode = st.sidebar.radio("Contact students by", ["Number of conversations", "Risk cut-off"])
if mode == "Number of conversations":
    value = st.sidebar.slider("Conversations", 0, 200, 40)
else:
    value = st.sidebar.slider("Contact students with a risk of at least", 0.02, 0.90, 0.20, 0.01)

st.title("Who should the study office talk to?")
st.caption("Week 6 · risk of leaving, from a model trained on the 2023 and 2024 cohorts and checked on 2025")

tab1, tab2, tab3, tab4 = st.tabs(["This week's list", "Mistakes of the rule", "Per group", "What it costs"])

# ---- 1. this week's list ----
with tab1:
    capacity = value if mode == "Number of conversations" else int((new["risk"] >= value).sum())
    ranked = new.sort_values("risk", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked["contact"] = np.where(ranked["rank"] <= capacity, "✅ yes", "")
    st.write(f"**{capacity}** of {len(ranked)} students are marked for a conversation with the current rule.")
    show = ["rank", "contact", "student_id", "risk", "programme", "international", "fees_owed",
            "submitted_share", "missed_last3", "quiz_mean", "logins_last3"]
    st.dataframe(ranked[show].style.format({"risk": "{:.0%}", "submitted_share": "{:.0%}",
                                            "quiz_mean": "{:.0f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("The list is a starting point for a conversation, not a verdict on a student. "
               "Look at why a student is on it before you call.")

# ---- 2. mistakes of the rule (2025 cohort) ----
flag = flagged_by(val, mode, value)
with tab2:
    tp, fp, fn, tn, precision, recall = boxes(val, flag)
    st.write(f"On last year's students ({len(val)}, of whom {int(val['left'].sum())} left), "
             f"this rule would have contacted **{tp + fp}**:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reached in time", tp)
    c2.metric("Worried for nothing", fp)
    c3.metric("Missed", fn)
    st.write(f"**{tp}** students reached in time, **{fp}** worried for nothing, **{fn}** missed.")
    st.write(f"Of the students contacted, **{precision:.0%}** were really at risk (precision). "
             f"Of the students who left, **{recall:.0%}** were reached (recall).")

# ---- 3. per group ----
with tab3:
    rows = []
    for g, name in [(0, "Domestic"), (1, "International")]:
        part = val[val["international"] == g]
        tp, fp, fn, tn, precision, recall = boxes(part, flag[part.index])
        rows.append({"Group": name, "Students": len(part), "Left": int(part["left"].sum()),
                     "Real share who left": part["left"].mean(), "Average predicted risk": part["risk"].mean(),
                     "Reached in time": tp, "Worried for nothing": fp, "Missed": fn,
                     "Precision": precision, "Recall": recall})
    st.dataframe(pd.DataFrame(rows).style.format({
        "Real share who left": "{:.1%}", "Average predicted risk": "{:.1%}",
        "Precision": "{:.0%}", "Recall": "{:.0%}"}), use_container_width=True, hide_index=True)
    st.caption("Same rule, same year, split by group. Few international students left, "
               "so differences between the groups are uncertain.")

# ---- 4. our own thing: what the rule costs ----
with tab4:
    st.write("These costs are assumptions. Change them and see how the best rule moves.")
    a, b, c, d = st.columns(4)
    talk = a.number_input("A conversation (DKK)", 0, 10_000, 500, 100)
    worry = b.number_input("A false alarm (DKK)", 0, 50_000, 2_000, 500)
    leave = c.number_input("A student who leaves (DKK)", 0, 500_000, 60_000, 5_000)
    helps = d.slider("Share a conversation keeps", 0.0, 1.0, 0.30, 0.05)

    def net(tp, fp):
        return tp * helps * leave - (tp + fp) * talk - fp * worry

    tp, fp, *_ = boxes(val, flag)
    st.metric("Net value of the current rule (2025 cohort)", f"{net(tp, fp):,.0f} DKK")

    cuts = np.round(np.arange(0.02, 0.91, 0.01), 2)
    values = [net(*boxes(val, val["risk"] >= cut)[:2]) for cut in cuts]
    best = cuts[int(np.argmax(values))]
    st.write(f"Best cut-off with these costs: **{best:.2f}**, which contacts "
             f"**{int((val['risk'] >= best).sum())}** students. "
             f"Break-even risk: {(talk + worry) / (helps * leave + worry):.2f}.")
    st.line_chart(pd.DataFrame({"Net value (DKK)": values}, index=cuts))
