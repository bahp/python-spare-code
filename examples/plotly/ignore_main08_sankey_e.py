# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:47:37 2025

@author: wlee-icare

Optimized version:
- Replaced slow, iterative flow-building loop with a fast, vectorized pandas merge.
- Improved efficiency of mapping items to classes.
- Used categorical data types to speed up group operations and reduce memory.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.cm import get_cmap
from typing import List, Set, Tuple, Dict, Optional
import random
import pathlib

# Read the merged dataset
cwd = os.getcwd()
print(cwd)

# Constants
DAY_MIN, DAY_MAX = 0, 7
ENABLE_COMBOS = True
COPRESCRIBE_WINDOW_MIN = 60
TIME_COL = "ORDER_DT_TM"
NEWS2_COL = "NEWS2_SCORE"

# Plotting settings
FIG_W, FIG_H = 26, 14
OUT_PNG = "aware_sankey_with_combos.png"
LW_BASE = 0.4
LW_K = 0.25
ALPHA = 0.5
X_PAD = 1.8

# ---------------- AWaRe lists -----------------
ACCESS_LIST = [
    "Amikacin", "Amoxicillin", "Amoxicillin (contains penicillin)", "Amoxicillin/clavulanic-acid",
    "Amoxiclavulanic-acid",
    "Benzylpenicillin sodium", "Co-amoxiclav", "Co-amoxiclav (contains penicillin)",
    "Co-amoxiclav (contains penicillin) (anes)", "Co-amoxiclav 400mg/57mg in 5ml oral suspension (contains penicillin)",
    "co-amoxiclav 250mg/62mg in 5ml oral suspension (contains penicillin)", "Cefalexin", "Cefazolin", "Chloramphenicol",
    "Clindamycin",
    "Doxycycline", "Flucloxacillin (contains penicillin)", "Gentamicin", "Gentamicin (anes)", "Mecillinam",
    "Metronidazole", "Metronidazole (anes)"
                     "Metronidazole_IV", "Metronidazole_oral", "Nitrofurantoin", "Co-amoxiclav in suspension",
    "Pivmecillinam", "Pivmecillinam (contains penicillin)", "Sulfamethoxazole/trimethoprim", "Trimethoprim",
    "co-trimoxazole"
]
WATCH_LIST = [
    "Azithromycin", "Cefepime", "Ceftazidime", "Ceftriaxone", "Cefuroxime", "Cefuroxime (anes)", "Ciprofloxacin",
    "Clarithromycin", "Ertapenem", "Erythromycin", "Fosfomycin", "Fosfomycin_oral", "Levofloxacin", "Meropenem",
    "Moxifloxacin", "Piperacillin/tazobactam", "piperacillin-tazobactam (contains penicillin)",
    "piperacillin + tazobactam (contains penicillin)",
    "Teicoplanin", "Teicoplanin (anes)", "Temocillin", "Temocillin (contains penicillin)", "Vancomycin",
    "Vancomycin (anes)", "Vancomycin (anes) 1g"
                         "Vancomycin_IV", "Vancomycin_oral"
]
RESERVE_LIST = [
    "Aztreonam", "Cefiderocol", "Ceftazidime/avibactam", "Ceftolozane/tazobactam", "Colistin", "Colistin_IV",
    "Dalbavancin",
    "Daptomycin", "Fosfomycin_IV", "Iclaprim", "Linezolid", "Tigecycline"
]


def norm(s: str) -> str:
    """
    Canonicalize antibiotic names.
    """
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\bcontains penicillin\b", "", s)
    s = re.sub(r"\b(iv|oral|po|im|sc|anes)\b", "", s)
    s = " ".join(tok for tok in re.split(r"[ \t]+", s) if not re.search(r"\d", tok))

    FORM_TERMS = {
        "susp", "susp.", "suspension", "solution", "infusion", "injection",
        "tablet", "tablets", "tab", "tabs", "cap", "caps", "capsule", "capsules",
        "powder", "vial", "ampoule", "amp", "syrup", "elixir", "cream", "ointment",
        "drops"
    }
    s = " ".join(tok for tok in s.split() if tok not in FORM_TERMS and tok != "in")
    s = re.sub(r"[/_+,]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    synonyms = {
        "co-amoxiclav": "co-amoxiclav", "amoxicillin clavulanic acid": "co-amoxiclav",
        "piperacillin tazobactam": "piperacillin-tazobactam",
        "cefuroxime axetil": "cefuroxime", "benzylpenicillin sodium": "benzylpenicillin",
        "amoxiclavulanic acid": "co-amoxiclav", "co trimoxazole": "co-trimoxazole",
        "ceftazidime avibactam": "ceftazidime/avibactam",
        "ceftolozane tazobactam": "ceftolozane/tazobactam"
    }
    # Simplified mapping: apply specific synonyms, otherwise return the cleaned string
    s_base = synonyms.get(s.replace(" ", "-"), s)  # Check for hyphenated version first
    return synonyms.get(s_base, s_base)


DRUG_TO_CLASS = {
    "benzylpenicillin": "Penicillin", "flucloxacillin": "Penicillin",
    'temocillin': "Penicillin", "amoxicillin": "Penicillin",
    "piperacillin-tazobactam": "Penicillin and beta-lactamase inhibitor",
    "pivmecillinam": "Penicillin",
    "co-amoxiclav": "Penicillin and beta-lactamase inhibitor",
    "cefalexin": "1st Generation Cephalosporin", "cefazolin": "1st Generation Cephalosporin",
    "cefuroxime": "2nd Generation Cephalosporin", "ceftriaxone": "3rd Generation Cephalosporin",
    "ceftazidime": "3rd Generation Cephalosporin", "cefepime": "4th Generation Cephalosporin",
    'cefiderocol': "5th Generation Cephalosporin", "clindamycin": "Lincomycin",
    "meropenem": "Carbapenem", "ertapenem": "Carbapenem",
    "aztreonam": "Monobactam", "gentamicin": "Aminoglycoside",
    "amikacin": "Aminoglycoside", "ciprofloxacin": "Fluoroquinolone",
    "levofloxacin": "Fluoroquinolone", "moxifloxacin": "Fluoroquinolone",
    "clarithromycin": "Macrolide", "erythromycin": "Macrolide",
    "azithromycin": "Macrolide", "vancomycin": "Glycopeptide",
    "teicoplanin": "Glycopeptide", "linezolid": "Oxazolidinone",
    "tigecycline": "Glycylcycline", "daptomycin": "Lipopeptide",
    "metronidazole": "Nitroimidazole", "doxycycline": "Tetracycline",
    "nitrofurantoin": "Nitrofuran", "fosfomycin": "Phosphonic acid",
    "co-trimoxazole": "Sulfonamides and trimethoprim",
    "trimethoprim": "Sulfonamides and trimethoprim", "chloramphenicol": "Amphenicol"
}


def lower_set(lst): return {norm(x) for x in lst}


ACCESS_N = lower_set(ACCESS_LIST)
WATCH_N = lower_set(WATCH_LIST)
RESERVE_N = lower_set(RESERVE_LIST)
AWARE_ALL = ACCESS_N | WATCH_N | RESERVE_N

CAT_RANK = {"Access": 1, "Watch": 2, "Reserve": 3}
RANK_TO_CAT = {v: k for k, v in CAT_RANK.items()}


def get_category(label: str) -> str:
    """AWaRe category; for combos like 'a+b' choose highest-risk component."""
    components = label.split("+")
    ranks = []
    for p in components:
        if p in ACCESS_N:
            ranks.append(CAT_RANK["Access"])
        elif p in WATCH_N:
            ranks.append(CAT_RANK["Watch"])
        elif p in RESERVE_N:
            ranks.append(CAT_RANK["Reserve"])
    return RANK_TO_CAT[max(ranks)] if ranks else None


CLASS_TO_CATEGORY = {drug_class: get_category(drug) for drug, drug_class in DRUG_TO_CLASS.items()}


def get_category_from_class(label: str) -> str:
    """AWaRe category from class; for combos like 'a+b' choose highest-risk component."""
    components = label.split("+")
    ranks = []
    for p in components:
        category = CLASS_TO_CATEGORY.get(p)
        if category: ranks.append(CAT_RANK[category])
    return RANK_TO_CAT[max(ranks)] if ranks else None


# ---------------------- creating synthetic data ----------------------
antibiotics = list(DRUG_TO_CLASS.keys())  # Use keys from a definitive mapping
n = 20000
np.random.seed(42)
random.seed(42)

subjects = np.random.randint(1000, 6000, size=n)
base_date = datetime(2021, 6, 20)
admission_dates = [base_date + timedelta(days=int(i % 10)) for i in range(n)]
news2_scores = np.random.randint(0, 20, size=n)
day_nums = np.random.randint(0, 7, size=n)
meds = np.random.choice(antibiotics, size=n)
order_times = [(admission_dates[i] + timedelta(hours=random.randint(0, 120))) for i in range(n)]

df = pd.DataFrame({
    "SUBJECT": subjects, "ADMISSION_DATE": admission_dates,
    "NEWS2_SCORE": news2_scores, "DAY_NUM": day_nums,
    "MEDICATION_NAME": meds, "ORDER_DT_TM": order_times,
})
df["ADMISSION_DATE"] = pd.to_datetime(df["ADMISSION_DATE"])
df["ORDER_DT_TM"] = pd.to_datetime(df["ORDER_DT_TM"])
df["drug_norm"] = df["MEDICATION_NAME"].map(norm).astype('category')
print(f"Synthetic dataset created with {len(df)} rows")

before = len(df)
df = df.drop_duplicates()
print(f"Exact duplicate rows removed: {before - len(df)}")

m = int(len(df) * 0.25)
extra = df.sample(m, replace=False, random_state=42).copy()
choices = np.array(antibiotics)


def pick_other(x):
    opts = choices[choices != x]
    return np.random.choice(opts) if len(opts) > 0 else x


extra["MEDICATION_NAME"] = extra["MEDICATION_NAME"].apply(pick_other)
extra["ORDER_DT_TM"] += pd.to_timedelta(np.random.randint(-45, 46, size=len(extra)), unit="m")
extra["drug_norm"] = extra["MEDICATION_NAME"].map(norm).astype('category')
df = pd.concat([df, extra], ignore_index=True)
print(f"\nSynthetic dataset augmented to {len(df)} rows to include combos\n")

df = df[df["DAY_NUM"].between(DAY_MIN, DAY_MAX)].dropna()
df = df.drop_duplicates(subset=["SUBJECT", "ADMISSION_DATE", "DAY_NUM", "drug_norm", TIME_COL, NEWS2_COL])
df = df[df["drug_norm"].isin(AWARE_ALL)].copy()

# --------------------------------------------------
# Combination rules
# --------------------------------------------------
COMBO_RULES: List[Tuple[str, str]] = []
TRIPLE_RULES: List[Tuple[Set[str], str]] = []


def add_pairs(bases: List[str], partners: List[str]):
    for b in bases:
        for p in partners:
            # Add rule as a sorted tuple to avoid duplicates like (a,b) and (b,a)
            COMBO_RULES.append(tuple(sorted((norm(b), norm(p)))))


X_list = ["co-amoxiclav", "ceftriaxone", "ciprofloxacin", "piperacillin-tazobactam", "meropenem", "temocillin"]
add_pairs(X_list, ["amikacin", "gentamicin"])
add_pairs(["co-amoxiclav", "ceftriaxone"], ["clarithromycin"])
vanco_partners = X_list + ["amikacin", "gentamicin", "clarithromycin"]
add_pairs(["vancomycin"], vanco_partners)
TRIPLE_RULES.append(
    ({"teicoplanin", "piperacillin-tazobactam", "amikacin"}, "teicoplanin+piperacillin-tazobactam+amikacin"))
add_pairs(["metronidazole"], ["cefuroxime", "ceftriaxone", "co-amoxiclav"])
add_pairs(["gentamicin"], ["amikacin"])
COMBO_RULES = sorted(list(set(COMBO_RULES)))  # Deduplicate and sort


# --------------------------------------------------
# Detect combos for a single patient-day
# --------------------------------------------------
def detect_combos_for_day(day_rows: pd.DataFrame, window_min: int = COPRESCRIBE_WINDOW_MIN) -> List[str]:
    drugs = day_rows[["drug_norm", TIME_COL]].drop_duplicates()
    if drugs["drug_norm"].nunique() < 2:
        return drugs["drug_norm"].unique().tolist()

    times = {d: sorted(drugs.loc[drugs["drug_norm"] == d, TIME_COL].dropna().tolist())
             for d in drugs["drug_norm"].unique()}
    used, items = set(), []

    # pairs
    for b, p in COMBO_RULES:
        if b in used or p in used: continue
        if b not in times or p not in times: continue

        matched = False
        for tb in times[b]:
            if any(abs((tb - tp).total_seconds()) <= window_min * 60 for tp in times[p]):
                matched = True;
                break
        if matched:
            items.append(f"{b}+{p}")
            used.update([b, p])

    # triple
    for drugset, label in TRIPLE_RULES:
        if any(d in used for d in drugset) or not drugset.issubset(times.keys()): continue

        # Simplified check: assume if all drugs present on same day, it's a combo
        items.append(label)
        used.update(drugset)

    # remaining singles
    items.extend([d for d in times if d not in used])
    return sorted(list(set(items)))


def assign_class_to_drug(item: str) -> str:
    """Maps a single drug or combo string to its corresponding class or combo class."""
    if "+" in item:
        return '+'.join(sorted([DRUG_TO_CLASS.get(drug, "Unknown") for drug in item.split('+')]))
    return DRUG_TO_CLASS.get(item, "Unknown")


# --------------------------------------------------
# Main plotting function with vectorized flow building
# --------------------------------------------------
def build_and_plot(df_in: pd.DataFrame, out_png: str, title_suffix: str = "",
                   normalize_widths: bool = False):
    # ---- Collapse to per-patient-per-day items ----
    daily_items = df_in.groupby(["SUBJECT", "ADMISSION_DATE", "DAY_NUM"]).apply(detect_combos_for_day)
    df_day = daily_items.explode().reset_index(name="item")
    df_day["item"] = df_day["item"].astype('category')
    df_day["category"] = df_day["item"].map(get_category).astype('category')

    # ---- OPTIMIZED: Vectorized flow building ----
    # 1. Create a map for all unique items to their class for efficiency
    unique_items = df_day["item"].unique()
    class_map = {item: assign_class_to_drug(item) for item in unique_items}
    df_day["item_class"] = df_day["item"].map(class_map)

    # 2. Prepare 'from' and 'to' DataFrames for merging
    df_from = df_day.rename(columns={'item_class': 'item_from', 'DAY_NUM': 'day_from'})
    df_to = df_day.rename(columns={'item_class': 'item_to', 'DAY_NUM': 'day_to'})
    df_to['day_from'] = df_to['day_to'] - 1  # Key to match with previous day

    # 3. Merge to create all-to-all transitions for each patient between consecutive days
    links_df = pd.merge(
        df_from[['SUBJECT', 'ADMISSION_DATE', 'day_from', 'item_from']],
        df_to[['SUBJECT', 'ADMISSION_DATE', 'day_from', 'item_to', 'day_to']],
        on=['SUBJECT', 'ADMISSION_DATE', 'day_from']
    )

    # 4. Aggregate to get flow counts
    flow = (links_df.groupby(["day_from", "item_from", "day_to", "item_to"], as_index=False)
            .size().rename(columns={"size": "count"}))
    flow = flow[flow['day_from'].between(DAY_MIN, DAY_MAX - 1)]  # Ensure within bounds

    # For prop-normalised widths
    if normalize_widths:
        denom = flow.groupby(["day_from", "day_to"])["count"].transform("sum")
        flow["prop"] = flow["count"] / denom.replace(0, np.nan)

    # ---- Ordering, colours, positions ----
    all_items = pd.concat([flow["item_from"], flow["item_to"]]).unique()
    categories = pd.Series({i: get_category_from_class(i) for i in all_items})

    items_access = sorted([i for i, cat in categories.items() if cat == "Access"])
    items_watch = sorted([i for i, cat in categories.items() if cat == "Watch"])
    items_reserve = sorted([i for i, cat in categories.items() if cat == "Reserve"])

    order = items_access + items_watch + items_reserve
    y_pos = {label: idx for idx, label in enumerate(order)}
    sep1 = len(items_access) - 0.5
    sep2 = sep1 + len(items_watch)

    palette = get_cmap("viridis", len(order))
    color_map = {label: palette(i) for i, label in enumerate(order)}
    x_pos = {d: (d - DAY_MIN) * X_PAD for d in range(DAY_MIN, DAY_MAX + 1)}

    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    def bezier(ax, x0, y0, x1, y1, lw, color, alpha):
        midx = (x0 + x1) / 2.0
        verts = [(x0, y0), (midx, y0), (midx, y1), (x1, y1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        patch = PathPatch(Path(verts, codes), facecolor='none', edgecolor=color, lw=lw, alpha=alpha)
        ax.add_patch(patch)

    for _, r in flow.sort_values("count", ascending=True).iterrows():
        if r["item_from"] not in y_pos or r["item_to"] not in y_pos: continue

        x0, y0 = x_pos[r["day_from"]], y_pos[r["item_from"]]
        x1, y1 = x_pos[r["day_to"]], y_pos[r["item_to"]]

        lw = LW_BASE + 6.0 * float(r.get("prop", 0.0)) if normalize_widths else LW_BASE + LW_K * float(r["count"])
        bezier(ax, x0, y0, x1, y1, lw, color_map.get(r["item_from"], "gray"), ALPHA)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylabel("Antibiotic Classes and Combinations")
    ax.set_xticks([x_pos[d] for d in range(DAY_MIN, DAY_MAX + 1)])
    ax.set_xticklabels([f"Day {d}" for d in range(DAY_MIN, DAY_MAX + 1)])
    ax.set_xlabel("Day")
    ax.set_xlim(x_pos[DAY_MIN] - 0.1 * X_PAD, x_pos[DAY_MAX] + 0.1 * X_PAD)
    ax.margins(x=0.05)

    ax.axhline(sep1, color='black', lw=1, linestyle='--', alpha=0.7)
    ax.axhline(sep2, color='black', lw=1, linestyle='--', alpha=0.7)

    ax.text(x_pos[DAY_MAX] + 0.2, sep1 / 2, "Access", va='center', ha='left', fontsize=12, fontweight='bold')
    ax.text(x_pos[DAY_MAX] + 0.2, (sep1 + sep2) / 2, "Watch", va='center', ha='left', fontsize=12, fontweight='bold')
    ax.text(x_pos[DAY_MAX] + 0.2, (sep2 + len(order)) / 2, "Reserve", va='center', ha='left', fontsize=12,
            fontweight='bold')

    ax.invert_yaxis()
    title = "Antibiotic Switching Between AWaRe Groups in Sepsis Patients"
    if title_suffix: title += f" ({title_suffix})"
    ax.set_title(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory

    return df_day, flow


# ==============================
# Summary helpers
# ==============================
def aware_mix_day0(df_day: pd.DataFrame) -> pd.Series:
    d0 = df_day[df_day["DAY_NUM"] == 0]
    mix = (d0["category"].value_counts(normalize=True)
           .reindex(["Access", "Watch", "Reserve"], fill_value=0.0) * 100).round(1)
    return mix


def top_items_day0(df_day: pd.DataFrame, n=10) -> pd.Series:
    d0 = df_day[df_day["DAY_NUM"] == 0]
    return (d0["item"].value_counts(normalize=True).head(n) * 100).round(1)


def escalation_stats(flow: pd.DataFrame) -> pd.Series:
    f = flow.copy()
    f["from_cat"] = f["item_from"].map(get_category_from_class)
    f["to_cat"] = f["item_to"].map(get_category_from_class)
    rank = {"Access": 1, "Watch": 2, "Reserve": 3}
    f["delta"] = f["to_cat"].map(rank) - f["from_cat"].map(rank)
    totals = f["count"].sum()
    if totals == 0:
        return pd.Series({"Escalate %": 0.0, "De-escalate %": 0.0, "Stable %": 0.0})
    esc = f.loc[f["delta"] > 0, "count"].sum() / totals * 100
    de = f.loc[f["delta"] < 0, "count"].sum() / totals * 100
    st = f.loc[f["delta"] == 0, "count"].sum() / totals * 100
    return pd.Series({"Escalate %": round(esc, 1), "De-escalate %": round(de, 1), "Stable %": round(st, 1)})


def print_summary(name: str, df_day: pd.DataFrame, flow: pd.DataFrame):
    print(f"\n==== {name} ====")
    print("Day-0 AWaRe mix (%):")
    print(aware_mix_day0(df_day).to_string())
    print("\nDay-0 top 10 items (% of cohort):")
    print(top_items_day0(df_day, n=10).to_string())
    print("\nDay 0→1 escalation/de-escalation (% of transitions):")
    print(escalation_stats(flow[flow['day_from'] == 0]).to_string())


# --------------------------------------------------
# Main execution block
# --------------------------------------------------

# Define the folder and file path
output_dir = pathlib.Path('objects/plot_main08_sankey/')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'synthetic_data.csv'
df.to_csv(output_file, index=False)
print(f"✅ DataFrame successfully saved to: {output_file}")

# Overall plot
print("\nProcessing: All patients...")
df_day_all, flow_all = build_and_plot(
    df, output_dir / "sankey_overall.png", "All patients", normalize_widths=False
)
print_summary("OVERALL", df_day_all, flow_all)

# NEWS2 ≥ 3
print("\nProcessing: Patients with NEWS2 >= 3...")
pat_ge3 = df.loc[(df["DAY_NUM"] == 0) & (df[NEWS2_COL] >= 3), "SUBJECT"].unique()
df_ge3 = df[df["SUBJECT"].isin(pat_ge3)].copy()
df_day_ge3, flow_ge3 = build_and_plot(
    df_ge3, output_dir / "sankey_NEWS2_ge3.png", "NEWS2 ≥3", normalize_widths=True
)
print_summary("NEWS2 ≥3", df_day_ge3, flow_ge3)

# NEWS2 ≥ 5
print("\nProcessing: Patients with NEWS2 >= 5...")
pat_ge5 = df.loc[(df["DAY_NUM"] == 0) & (df[NEWS2_COL] >= 5), "SUBJECT"].unique()
df_ge5 = df[df["SUBJECT"].isin(pat_ge5)].copy()
df_day_ge5, flow_ge5 = build_and_plot(
    df_ge5, output_dir / "sankey_NEWS2_ge5.png", "NEWS2 ≥5", normalize_widths=True
)
print_summary("NEWS2 ≥5", df_day_ge5, flow_ge5)

# ================== QC ==================
print("\n--- QC: Top Combo Items ---")
print("\nTop combos (OVERALL):")
print(df_day_all[df_day_all["item"].str.contains(r"\+", na=False)]["item"].value_counts().head(10))
print("\nTop combos (NEWS2 ≥3):")
print(df_day_ge3[df_day_ge3["item"].str.contains(r"\+", na=False)]["item"].value_counts().head(10))
print("\nTop combos (NEWS2 ≥5):")
print(df_day_ge5[df_day_ge5["item"].str.contains(r"\+", na=False)]["item"].value_counts().head(10))

print("\n✅ All processing complete.")
