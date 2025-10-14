# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:47:37 2025

@author: wlee-icare
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


def main():
    # Read the merged dataset
    cwd = os.getcwd()
    print(cwd)
    new_working_directory = r'\\ICH-FS-A\SharedData\SharedBatch01\Business Intelligence\iCARE - Research Project Folders\NIBDAPC Development of SMART PATH Sepsis Trial Protocol (130)\Tim\Winnie_data\GC4'
    #os.chdir(new_working_directory)
    print(os.path.exists(new_working_directory))

    # CSV_IN = "ABX_cohort_upto7days_from_final_newscohort.csv"
    DAY_MIN, DAY_MAX = 0, 7

    # Co-prescription detection
    ENABLE_COMBOS = True
    COPRESCRIBE_WINDOW_MIN = 60  # define co-prescription if drugs within ≤60 minutes
    TIME_COL = "ORDER_DT_TM"  # <-- my dataset has this column
    NEWS2_COL = "NEWS2_SCORE"

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
        Canonicalize antibiotic names:
          - lowercases
          - drops parentheticals (e.g., '(contains penicillin)', '(anes)')
          - removes route tags (iv, oral, po, im, sc, anes)
          - strips any tokens containing digits (dose/strength like 250mg/5ml)
           - remove formulation words (suspension/tablet/solution/infusion/etc.)
          - normalizes separators, collapses whitespace
          - maps common synonyms to a canonical name
        """
        if pd.isna(s):
            return ""
        s = str(s).lower().strip()

        # remove any parenthetical note entirely
        s = re.sub(r"\([^)]*\)", "", s)
        # explicit phrase
        s = re.sub(r"\bcontains penicillin\b", "", s)

        # remove route/administration tags
        s = re.sub(r"\b(iv|oral|po|im|sc|anes)\b", "", s)

        # remove tokens with digits (dose/volume/strength)
        s = " ".join(tok for tok in re.split(r"[ \t]+", s) if not re.search(r"\d", tok))

        # 4) remove formulation terms (this is what was missing before)
        FORM_TERMS = {
            "susp", "susp.", "suspension", "solution", "infusion", "injection",
            "tablet", "tablets", "tab", "tabs", "cap", "caps", "capsule", "capsules",
            "powder", "vial", "ampoule", "amp", "syrup", "elixir", "cream", "ointment",
            "drops"
        }
        s = " ".join(tok for tok in s.split() if tok not in FORM_TERMS and tok != "in")

        # normalize separators to spaces
        s = re.sub(r"[/_+,]", " ", s)
        s = s.replace("-", " ")

        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()

        # synonym map (after cleaning)
        synonyms = {
            # co-amoxiclav and amoxi family
            "co-amoxiclav": "co-amoxiclav",
            "amoxicillin clavulanic acid": "co-amoxiclav",
            "co-amoxiclav (contains penicillin)": "co-amoxiclav",
            "co-amoxiclav (contains penicillin) (anes)": "co-amoxiclav",
            "co-amoxiclav 400mg/57mg in 5ml oral suspension (contains penicillin)": "co-amoxiclav",
            "co-amoxiclav 250mg/62mg in 5ml oral suspension (contains penicillin)": "co-amoxiclav",
            "co-amoxiclav 250mg/62mg in 5ml oral suspension (contains penicillin)": "co-amoxiclav",
            "amoxicillin (contains penicillin)": "amoxicillin",
            "co-amoxiclav in suspension": "co-amoxiclav",

            # piperacillin-tazobactam family
            "piperacillin-tazobactam (contains penicillin)": "piperacillin-tazobactam",
            "piperacillin + tazobactam (contains penicillin)": "piperacillin-tazobactam",

            # cefuroxime family
            "cefuroxime": "cefuroxime",
            "cefuroxime axetil": "cefuroxime",
            "cefuroxime (anes)": "cefuroxime",

            # vanc family
            "vancomycin": "vancomycin",
            "vancomycin (anes)": "vancomycin",
            "vancomycin (anes) 1g": "vancomycin",

            # others family
            "metronidazole (anes)": "metronidazole",
            "teicoplanin (anes)": "teicoplanin",
            "gentamicin (anes)": "gentamicin",
            "pivmecillinam (contains penicillin)": "pivmecillinam",

            # straightforward names (routes stripped already)
            "vancomycin": "vancomycin",
            "metronidazole": "metronidazole",
            "teicoplanin": "teicoplanin",
            "gentamicin": "gentamicin",
            "amikacin": "amikacin",
            "ciprofloxacin": "ciprofloxacin",
            "ceftriaxone": "ceftriaxone",
            "temocillin": "temocillin",
            "meropenem": "meropenem",
            "mecillinam": "mecillinam",
            "pivmecillinam": "pivmecillinam",
            "flucloxacillin": "flucloxacillin",
            "nitrofurantoin": "nitrofurantoin",
            "chloramphenicol": "chloramphenicol",
            "clindamycin": "clindamycin",
            "azithromycin": "azithromycin",
            "clarithromycin": "clarithromycin",
            "erythromycin": "erythromycin",
            "levofloxacin": "levofloxacin",
            "moxifloxacin": "moxifloxacin",
            "cefazolin": "cefazolin",
            "cefalexin": "cefalexin",
            "ceftazidime": "ceftazidime",
            "ceftazidime avibactam": "ceftazidime/avibactam",
            "ceftolozane tazobactam": "ceftolozane/tazobactam",
            "aztreonam": "aztreonam",
            "linezolid": "linezolid",
            "tigecycline": "tigecycline",
            "daptomycin": "daptomycin",
            "fosfomycin": "fosfomycin",
            "fosfomycin iv": "fosfomycin_iv",
            "fosfomycin oral": "fosfomycin",  # collapse to base
            "benzylpenicillin sodium": "benzylpenicillin",
            "amoxicillin": "amoxicillin",
            "amoxiclavulanic acid": "co-amoxiclav",
            "co-trimoxazole": "co-trimoxazole",
            "trimethoprim": "trimethoprim",
        }
        return synonyms.get(s, s)


    DRUG_TO_CLASS = {
        "benzylpenicillin": "Penicillin",
        "flucloxacillin": "Penicillin",
        'temocillin': "Penicillin",
        "amoxicillin": "Penicillin",
        "piperacillin tazobactam": "Penicillin and beta-lactamase inhibitor",
        "pivmecillinam": "Penicillin",
        "co-amoxiclav": "Penicillin and beta-lactamase inhibitor",
        "cefalexin": "1st Generation Cephalosporin",
        "cefazolin": "1st Generation Cephalosporin",
        "cefuroxime": "2nd Generation Cephalosporin",
        "ceftriaxone": "3rd Generation Cephalosporin",
        "ceftazidime": "3rd Generation Cephalosporin",
        "cefepime": "4th Generation Cephalosporin",
        'cefiderocol': "5th Generation Cephalosporin",
        "clindamycin": "Lincomycin",
        "meropenem": "Carbapenem",
        "ertapenem": "Carbapenem",
        "aztreonam": "Monobactam",
        "gentamicin": "Aminoglycoside",
        "amikacin": "Aminoglycoside",
        "ciprofloxacin": "Fluoroquinolone",
        "levofloxacin": "Fluoroquinolone",
        "moxifloxacin": "Fluoroquinolone",
        "clarithromycin": "Macrolide",
        "erythromycin": "Macrolide",
        "azithromycin": "Macrolide",
        "vancomycin": "Glycopeptide",
        "teicoplanin": "Glycopeptide",
        "linezolid": "Oxazolidinone",
        "tigecycline": "Glycylcycline",
        "daptomycin": "Lipopeptide",
        "metronidazole": "Nitroimidazole",
        "doxycycline": "Tetracycline",
        "nitrofurantoin": "Nitrofuran",
        "fosfomycin": "Phosphonic acid",
        "co trimoxazole": "Sulfonamides and trimethoprim",
        "trimethoprim": "Sulfonamides and trimethoprim",
        "chloramphenicol": "Amphenicol"
    }

    DRUG_TO_CLASS.update({k.replace(" ", "-"): v for k, v in list(DRUG_TO_CLASS.items())})
    DRUG_TO_CLASS.update({k.replace("-", " "): v for k, v in list(DRUG_TO_CLASS.items())})


    def lower_set(lst): return {norm(x) for x in lst}


    ACCESS_N = lower_set(ACCESS_LIST)
    WATCH_N = lower_set(WATCH_LIST)
    RESERVE_N = lower_set(RESERVE_LIST)
    AWARE_ALL = ACCESS_N | WATCH_N | RESERVE_N

    CAT_RANK = {"Access": 1, "Watch": 2, "Reserve": 3}
    RANK_TO_CAT = {v: k for k, v in CAT_RANK.items()}


    def get_category(label: str) -> str:
        """AWaRe category; for combos like 'a+b' choose highest-risk component."""
        if "+" in label:
            ranks = []
            for p in label.split("+"):
                if p in ACCESS_N:
                    ranks.append(CAT_RANK["Access"])
                elif p in WATCH_N:
                    ranks.append(CAT_RANK["Watch"])
                elif p in RESERVE_N:
                    ranks.append(CAT_RANK["Reserve"])
            return RANK_TO_CAT[max(ranks)] if ranks else None
        if label in ACCESS_N: return "Access"
        if label in WATCH_N:  return "Watch"
        if label in RESERVE_N: return "Reserve"
        return None


    CLASS_TO_CATEGORY = {drug_class: get_category(drug) for drug, drug_class in DRUG_TO_CLASS.items()}


    def get_category_from_class(label: str) -> str:
        """AWaRe category; for combos like 'a+b' choose highest-risk component."""
        if "+" in label:
            ranks = []
            for p in label.split("+"):
                if CLASS_TO_CATEGORY[p] == 'Access':
                    ranks.append(CAT_RANK["Access"])
                elif p in CLASS_TO_CATEGORY[p] == 'Watch':
                    ranks.append(CAT_RANK["Watch"])
                elif p in CLASS_TO_CATEGORY[p] == 'Reserve':
                    ranks.append(CAT_RANK["Reserve"])
            return RANK_TO_CAT[max(ranks)] if ranks else None
        else:
            return CLASS_TO_CATEGORY[label]
        return None


    # ---------------------- creating synthetic data ----------------------
    # usecols = ["SUBJECT","ADMISSION_DATE","DAY_NUM","MEDICATION_NAME",TIME_COL,NEWS2_COL]
    # df = pd.read_csv(CSV_IN, usecols=usecols,
    #                 parse_dates=["ADMISSION_DATE", TIME_COL],
    #                 low_memory=False)

    import random
    import numpy as np
    from datetime import datetime, timedelta

    antibiotics = [
        "amoxicillin", "co-trimoxazole", "meropenem",
        "metronidazole", "amikacin", "cefuroxime", "ciprofloxacin",
        "piperacillin-tazobactam", "vancomycin", "gentamicin",
        "flucloxacillin", "teicoplanin", "ceftriaxone", "temocillin", "fosfomycin",
        "clarithromycin", "aztreonam", "flucloxacillin", "gentamicin",
        "linezolid", "erythromycin", "ciprofloxacin", "ceftriaxone", "cefepime",
        "ceftazidime", "pivmecillinam", "chloramphenicol", "tazocin", "co-amoxiclav",
        "doxycycline"
    ]

    n = 20000
    np.random.seed(42)
    random.seed(42)

    subjects = np.random.randint(1000, 6000, size=n)
    base_date = datetime(2021, 6, 20)
    admission_dates = [base_date + timedelta(days=int(i % 10)) for i in range(n)]
    news2_scores = np.random.randint(0, 20, size=n)
    day_nums = np.random.randint(0, 7, size=n)
    meds = np.random.choice(antibiotics, size=n)
    order_times = [
        (admission_dates[i] + timedelta(hours=random.randint(0, 120))).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]
    drug_norms = [m.lower().replace(" ", "-") for m in meds]

    df = pd.DataFrame({
        "SUBJECT": subjects,
        "ADMISSION_DATE": [d.strftime("%Y-%m-%d %H:%M:%S") for d in admission_dates],
        "NEWS2_SCORE": news2_scores,
        "DAY_NUM": day_nums,
        "MEDICATION_NAME": meds,
        "ORDER_DT_TM": order_times,
        "drug_norm": drug_norms
    })

    df["drug_norm"] = df["MEDICATION_NAME"].map(norm)

    df["ORDER_DT_TM"] = pd.to_datetime(df["ORDER_DT_TM"])
    df["ADMISSION_DATE"] = pd.to_datetime(df["ADMISSION_DATE"])

    print(f"Synthetic dataset created with {len(df)} rows")
    print(df.head())

    before = len(df)
    df = df.drop_duplicates()
    print(f"Exact duplicate rows removed: {before - len(df)}")

    # df = df[df["DAY_NUM"].between(DAY_MIN, DAY_MAX)].dropna(subset=usecols).copy()

    # getting same-day prescriptions so combos do apopear in synthetic data
    m = int(len(df) * 0.25)
    extra = df.sample(m, replace=False).copy()

    choices = np.array(antibiotics)


    def pick_other(x):
        opts = choices[choices != x]
        return np.random.choice(opts) if len(opts) else x


    extra["MEDICATION_NAME"] = extra["MEDICATION_NAME"].apply(pick_other)
    # nearby time within +/- 45 minutes
    extra["ORDER_DT_TM"] = extra["ORDER_DT_TM"] + pd.to_timedelta(
        np.random.randint(-45, 46, size=len(extra)), unit="m"
    )
    extra["drug_norm"] = extra["MEDICATION_NAME"].map(norm)

    df = pd.concat([df, extra], ignore_index=True)

    print(df.head())
    print(f"\nSynthetic dataset created with {len(df)} rows\n")

    before = len(df)
    df = df[df["DAY_NUM"].between(DAY_MIN, DAY_MAX)].dropna(
        subset=["SUBJECT", "ADMISSION_DATE", "DAY_NUM", "MEDICATION_NAME", TIME_COL, NEWS2_COL])
    df["drug_norm"] = df["MEDICATION_NAME"].map(norm)
    before2 = len(df)
    df = df.drop_duplicates(subset=["SUBJECT", "ADMISSION_DATE", "DAY_NUM", "drug_norm", TIME_COL, NEWS2_COL])
    print(f"Per patient-day-drug duplicates removed: {before2 - len(df)}")

    df = df[df["drug_norm"].isin(AWARE_ALL)].copy()

    # --------------------------------------------------
    # Combination rules (exactly as requested)
    # --------------------------------------------------
    COMBO_RULES: List[Tuple[str, str]] = []
    TRIPLE_RULES: List[Tuple[Set[str], str]] = []


    def add_pairs(bases: List[str], partners: List[str]):
        for b in bases:
            for p in partners:
                COMBO_RULES.append((norm(b), norm(p)))


    # 1) X + (amikacin | gentamicin)
    X_list = ["co-amoxiclav", "ceftriaxone", "ciprofloxacin",
              "piperacillin-tazobactam", "meropenem", "temocillin"]
    add_pairs(X_list, ["amikacin", "gentamicin"])

    # 2) co-amoxiclav + clarithromycin; ceftriaxone + clarithromycin
    add_pairs(["co-amoxiclav", "ceftriaxone"], ["clarithromycin"])

    # 3) vancomycin + any of the above (+ amikacin/gentamicin/clarithro)
    vanco_partners = ["co-amoxiclav", "ceftriaxone", "ciprofloxacin",
                      "piperacillin-tazobactam", "meropenem", "temocillin",
                      "amikacin", "gentamicin", "clarithromycin"]
    add_pairs(["vancomycin"], vanco_partners)

    # 4) TRIPLE: teicoplanin + piperacillin-tazobactam + amikacin
    TRIPLE_RULES.append(({"teicoplanin", "piperacillin-tazobactam", "amikacin"},
                         "teicoplanin+piperacillin-tazobactam+amikacin"))

    # 5) metronidazole + (cefuroxime | ceftriaxone | co-amoxiclav)
    add_pairs(["metronidazole"], ["cefuroxime", "ceftriaxone", "co-amoxiclav"])

    # 6) gentamicin + amikacin (optional)
    add_pairs(["gentamicin"], ["amikacin"])


    # --------------------------------------------------
    # Detect combos for a single patient-day
    # --------------------------------------------------
    def detect_combos_for_day(day_rows: pd.DataFrame,
                              window_min: int = COPRESCRIBE_WINDOW_MIN,
                              collapse: bool = True) -> List[str]:
        drugs = day_rows[["drug_norm", TIME_COL]].drop_duplicates()
        if drugs["drug_norm"].nunique() < 2:
            return drugs["drug_norm"].unique().tolist()

        # map drug -> sorted timestamps (empty list if NaT)
        times = {d: sorted(drugs.loc[drugs["drug_norm"] == d, TIME_COL].dropna().tolist())
                 for d in drugs["drug_norm"].unique()}

        used, items = set(), []

        # pairs
        for b, p in COMBO_RULES:
            if b not in times or p not in times: continue
            if collapse and (b in used or p in used): continue

            matched = False
            tb_list, tp_list = times[b], times[p]
            if tb_list and tp_list:
                for tb in tb_list:
                    if any(abs((tb - tp).total_seconds()) <= window_min * 60 for tp in tp_list):
                        matched = True;
                        break
            else:
                matched = True  # accept same-day if one time missing

            if matched:
                items.append("+".join(sorted([b, p])))
                if collapse: used.update([b, p])

        # triple (explicit)
        for drugset, label in TRIPLE_RULES:
            if not drugset.issubset(times.keys()): continue
            if collapse and any(d in used for d in drugset): continue

            ok = False
            anchors = times[next(iter(drugset))] or [None]
            for t0 in anchors:
                if t0 is None: ok = True; break
                close_all = True
                for d in (drugset - {next(iter(drugset))}):
                    if not times[d] or not any(abs((t0 - td).total_seconds()) <= window_min * 60 for td in times[d]):
                        close_all = False;
                        break
                if close_all: ok = True; break
            if ok:
                items.append(label)
                if collapse: used.update(drugset)

        # remaining singles
        for d in times.keys():
            if collapse and (d in used): continue
            items.append(d)

        return sorted(set(items))


    def assign_class_to_drug(a: str) -> str:
        '''
        Group individual drugs into a class using a dictionary.

        If there is a combination use the 'worst' one

        Parameters
        ----------
        a : str
            The individual drug.

        Returns
        -------
        str
            The class.

        '''
        if a in DRUG_TO_CLASS.keys():
            # get classes
            drug_class = DRUG_TO_CLASS[a]
            # get (importance) categories for the drugs in prescription
            # drug_category = get_category(a)
        # we have more than one drug in the prescription
        else:
            drugs_in_pres = a.split('+')
            # get list of classes
            drug_class = '+'.join([DRUG_TO_CLASS[drug] for drug in drugs_in_pres])
            # # get (importance) categories for the drugs in prescription
            # drug_category = [get_category(drug) for drug in drugs_in_pres]
            # drug_category = get_most_important_category(drug_category)

        return drug_class


    # --------------------------------------------------
    # Collapse to per-day items and build flows day→day
    # --------------------------------------------------
    def build_and_plot(df_in: pd.DataFrame, out_png: str, title_suffix: str = "",
                       normalize_widths: bool = False):
        # ---- collapse to per-patient-per-day items ----
        rows_collapsed = []
        for (sid, adm), grp in df_in.groupby(["SUBJECT", "ADMISSION_DATE"]):
            for d in range(DAY_MIN, DAY_MAX + 1):
                day = grp[grp["DAY_NUM"] == d]
                items = detect_combos_for_day(day, collapse=True)
                for it in items:
                    rows_collapsed.append((sid, adm, d, it))

        df_day = pd.DataFrame(rows_collapsed, columns=["SUBJECT", "ADMISSION_DATE", "DAY_NUM", "item"])
        df_day["category"] = df_day["item"].map(get_category)

        links = []
        for (sid, adm), grp in df_day.groupby(["SUBJECT", "ADMISSION_DATE"]):
            day_map = {k: v["item"].tolist() for k, v in grp.groupby("DAY_NUM")}
            for d in range(DAY_MIN, DAY_MAX):
                if d in day_map and (d + 1) in day_map:
                    for a in day_map[d]:
                        # 'a' is the drug on day d, b is the drug on day d+1, lookup the 'a' in a dict
                        # and change to the 'class'
                        from_class = assign_class_to_drug(a)
                        for b in day_map[d + 1]:
                            to_class = assign_class_to_drug(b)
                            links.append((from_class, d, d + 1, to_class))

        flow = (pd.DataFrame(links, columns=["item_from", "day_from", "day_to", "item_to"])
                .groupby(["day_from", "item_from", "day_to", "item_to"], as_index=False)
                .size().rename(columns={"size": "count"}))

        # For prop-normalised widths
        if normalize_widths:
            denom = flow.groupby(["day_from", "day_to"])["count"].transform("sum").replace(0, np.nan)
            flow["prop"] = flow["count"] / denom

        # --------------------------------------------------
        # Ordering, colours, positions
        # --------------------------------------------------
        totals = pd.Series(pd.concat([flow["item_from"], flow["item_to"]])).value_counts()
        items_access = [i for i in totals.index if get_category_from_class(i) == "Access"]
        items_watch = [i for i in totals.index if get_category_from_class(i) == "Watch"]
        items_reserve = [i for i in totals.index if get_category_from_class(i) == "Reserve"]

        order = items_access + items_watch + items_reserve
        order = list(dict.fromkeys(order))  # dedup preserve order
        y_pos = {label: idx for idx, label in enumerate(order)}

        sep1 = len(items_access) - 0.5
        sep2 = len(items_access) + len(items_watch) - 0.5

        palette = plt.cm.get_cmap("tab20", len(order))  # discrete colormap with N colors
        color_map = {label: palette(i) for i, label in enumerate(order)}

        x_pos = {d: (d - DAY_MIN) * X_PAD for d in range(DAY_MIN, DAY_MAX + 1)}

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        def bezier(ax, x0, y0, x1, y1, lw, color, alpha):
            """
            Smooth S-curve from (x0,y0) to (x1,y1) using a single cubic Bezier.
            Control points are vertically aligned at the midpoint in x.
            """
            midx = (x0 + x1) / 2.0
            verts = [(x0, y0), (midx, y0), (midx, y1), (x1, y1)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            patch = PathPatch(
                Path(verts, codes),
                facecolor='none',
                edgecolor=color,
                lw=lw,
                alpha=alpha,
                capstyle='butt',  # avoid overhang before Day 0
                joinstyle='round',
                clip_on=True
            )
            ax.add_patch(patch)

        for _, r in flow.sort_values("count", ascending=True).iterrows():
            if r["item_from"] not in y_pos or r["item_to"] not in y_pos:
                continue
            x0, y0 = x_pos[r["day_from"]], y_pos[r["item_from"]]
            x1, y1 = x_pos[r["day_to"]], y_pos[r["item_to"]]
            if normalize_widths:
                lw = LW_BASE + 6.0 * float(r.get("prop", 0.0))
            else:
                lw = LW_BASE + LW_K * float(r["count"])
            bezier(ax, x0, y0, x1, y1, lw, color_map[r["item_from"]], ALPHA)

        # y-axis labels
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_ylabel("Antibiotics and combinations\nin each AWaRe group")

        # x-axis labels + exact limits + Day-0 spine (fixes left overhang)
        ax.set_xticks([x_pos[d] for d in range(DAY_MIN, DAY_MAX + 1)])
        ax.set_xticklabels([f"Day {d}" for d in range(DAY_MIN, DAY_MAX + 1)])
        ax.set_xlabel("Day")

        ax.set_xlim(-0.05 * X_PAD, x_pos[DAY_MAX] + 0.05 * X_PAD)
        ax.margins(x=0.05)
        ax.axvline(x_pos[DAY_MIN], color="0.6", lw=0.6, alpha=0.8, zorder=5)

        # Group separators & labels
        ax.axhline(sep1, color='black', lw=1, linestyle='--', alpha=0.7)
        ax.axhline(sep2, color='black', lw=1, linestyle='--', alpha=0.7)
        ax.text(x_pos[DAY_MAX] + 0.2, (0 + max(sep1, 0)) / 2, "Access", va='center', ha='left', fontsize=12,
                fontweight='bold')
        ax.text(x_pos[DAY_MAX] + 0.2, (max(sep1, 0) + max(sep2, 0)) / 2, "Watch", va='center', ha='left', fontsize=12,
                fontweight='bold')
        ax.text(x_pos[DAY_MAX] + 0.2, (sep2 + len(order)) / 2, "Reserve", va='center', ha='left', fontsize=12,
                fontweight='bold')

        # Vertical separators between days
        for d in range(DAY_MIN + 1, DAY_MAX):
            ax.axvline((x_pos[d] + x_pos[d + 1]) / 2, color='grey', lw=0.5, alpha=0.5)

        ax.invert_yaxis()
        title = "Antibiotic switching between AWaRE groups in sepsis patients"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
        #plt.show()

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
        s = (d0["item"].value_counts(normalize=True).head(n) * 100).round(1)
        return s


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
        return pd.Series({"Escalate %": round(esc, 1),
                          "De-escalate %": round(de, 1),
                          "Stable %": round(st, 1)})


    def print_summary(name: str, df_day: pd.DataFrame, flow: pd.DataFrame):
        print(f"\n==== {name} ====")
        mix = aware_mix_day0(df_day)
        print("Day-0 AWaRe mix (%):")
        print(mix.to_string())
        print("\nDay-0 top 10 items (% of cohort):")
        print(top_items_day0(df_day, n=10).to_string())
        print("\nDay 0→1 escalation/de-escalation (% of transitions):")
        print(escalation_stats(flow).to_string())


    # --------------------------------------------------
    # Build cohorts by NEWS2 at admission and plot
    # --------------------------------------------------

    # Patients with NEWS2 >= 3 at Day 0
    OUT_PREFIX = "news3_sankey"
    OUT_PREFIX2 = "news5_sankey"

    # Overall
    df_day_all, flow_all = build_and_plot(
        df, f"{OUT_PREFIX}_overall.png", "All patients", normalize_widths=False
    )
    print_summary("OVERALL", df_day_all, flow_all)

    # NEWS2 ≥ 3
    pat_ge3 = df.loc[(df["DAY_NUM"] == 0) & (df[NEWS2_COL] >= 3), "SUBJECT"].unique()
    df_ge3 = df[df["SUBJECT"].isin(pat_ge3)].copy()
    df_day_ge3, flow_ge3 = build_and_plot(  # <-- UNPACK BOTH
        df_ge3, f"{OUT_PREFIX}_NEWS2_ge3.png", "NEWS2 ≥3", normalize_widths=True
    )
    print_summary("NEWS2 ≥3", df_day_ge3, flow_ge3)

    # NEWS2 ≥ 5
    pat_ge5 = df.loc[(df["DAY_NUM"] == 0) & (df[NEWS2_COL] >= 5), "SUBJECT"].unique()
    df_ge5 = df[df["SUBJECT"].isin(pat_ge5)].copy()
    df_day_ge5, flow_ge5 = build_and_plot(  # <-- UNPACK BOTH
        df_ge5, f"{OUT_PREFIX}_NEWS2_ge5.png", "NEWS2 ≥5", normalize_widths=True
    )
    print_summary("NEWS2 ≥5", df_day_ge5, flow_ge5)

    # ================== QC ==================
    # Optional QC
    print("\nTop combos (OVERALL):")
    print(df_day_all[df_day_all["item"].str.contains(r"\+")]["item"].value_counts().head(15))
    print("\nTop combos (NEWS2 ≥3):")
    print(df_day_ge3[df_day_ge3["item"].str.contains(r"\+")]["item"].value_counts().head(15))
    print("\nTop combos (NEWS2 ≥5):")
    print(df_day_ge5[df_day_ge5["item"].str.contains(r"\+")]["item"].value_counts().head(15))

# ----------------- PROFILING EXECUTION -----------------
if __name__ == "__main__":

    import cProfile
    import pstats
    import io

    # ----------------- PROFILING SETUP -----------------
    # Create a profiler object
    profiler = cProfile.Profile()

    print("Starting script with profiling...")
    profiler.enable()

    main()  # Run the main logic

    profiler.disable()
    print("\n\n--- PROFILING RESULTS ---")

    # Create a stream to capture the stats output
    s = io.StringIO()
    # Sort stats by cumulative time and print the top 30 functions
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    stats.print_stats(30)
    print(s.getvalue())
    print("--- END OF PROFILING RESULTS ---\n")