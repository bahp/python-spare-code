"""
Threshold moving
----------------

 .. note::  https://en.wikipedia.org/wiki/Sensitivity_and_specificity
"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Libraries scikits
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split


def display_npv_ppv_curve(ppv, npv, ths, idx):
    """This method plots the curve

    Parameters
    ----------
    ppv: array-like
    npv: array-like
    ths: array-like
    idx: integer
    """
    # Display
    f, axes = plt.subplots(1, 1)
    axes.plot(ths, npv, marker='o', label='npv')
    axes.plot(ths, ppv, marker='o', label='ppv')
    axes.set(aspect='equal', xlim=[0,1], ylim=[0,1],
        xlabel='threshold', title='th={0}, npv={1}, ppv={2}' \
            .format(round(ths[idx], 3),
                    round(npv[idx], 3),
                    round(ppv[idx], 3)))
    plt.legend()


def npv_ppv_from_sens_spec(sens, spec, prev):
    """Compute npv and ppv.

    Parameters
    ----------
    sens: array-like
    spec: array-like
    prev: float
    """
    npv = (spec * (1 - prev)) / ((spec * (1 - prev)) + ((1 - sens) * prev))
    ppv = (sens * prev) / ((sens * prev) + ((1 - spec) * (1 - prev)))
    return npv, ppv



# ----------------------
# Load data
# ----------------------
# Fetch data
X, y = fetch_openml(data_id=1464,
                    return_X_y=True,
                    as_frame=True)

# Format y to binary (0,1)
y = y.replace({'1':0, '2':1})

# Split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y)

# ----------------------
# Create pipeline
# ----------------------
# Create pipeline
clf = make_pipeline(
    StandardScaler(),
    #LogisticRegression(random_state=0)
    ExtraTreesClassifier(n_estimators=100)
)

# Train
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# .. note: Some classifiers do not have the decision
#          function method but all implement the
#          predict_proba.
#y_score = clf.decision_function(X_test)

# -----------------------
# Show confusion matrix
# -----------------------
# .. note: We are using Display objects to plot
#          the graphs, they could also be displayed
#          using the functions or matplotlib
#          directly.
#
# plot_roc_curve(clf, X_test, y_test, ax=ax_roc, name=name)
# plot_det_curve(clf, X_test, y_test, ax=ax_roc, name=name)

# Libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

# Prevalence
prev = np.sum(y_test) / len(y_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# .. note: It is possible to use either y_score
#          or y_prob in the roc_curve function
# .. note: sens=tpr, spec=1-fpr
# Compute ROC curve
fpr, tpr, ths1 = roc_curve(
    y_test, y_prob[:, 1],
    drop_intermediate=False)

# .. note: ppv=prec, sens=recall
# Compute PR curve
prec, recall, ths2 = \
    precision_recall_curve(y_test, y_prob[:, 1])

# Create plot objects
cm_display = ConfusionMatrixDisplay(cm)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

# Create figure
f, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.flatten()

# Display
cm_display.plot()
roc_display.plot(ax=axes[0])
pr_display.plot(ax=axes[1])

# Configure
for ax in axes:
    ax.set(aspect='equal', xlim=[0,1], ylim=[0,1])
plt.tight_layout()



# ---------
# Option I
# ---------
# Compute the npv and ppv from the sensitivity
# and specificity values obtained from the
# 'roc_curve' function.

# Compute ROC curve
fpr, tpr, ths1 = roc_curve(
    y_test, y_prob[:, 1],
    drop_intermediate=False)

# Compute npv and ppv
npv, ppv = npv_ppv_from_sens_spec( \
    sens=tpr, spec=1-fpr, prev=prev)

# Create DataFrame
results = pd.DataFrame(
    data=np.array([ths1, ppv, npv, tpr, 1-fpr]).T,
    columns=['th', 'ppv', 'npv', 'sens', 'spec']
).sort_values(by='th')

# Add gmean
results['gmean'] = np.sqrt(tpr * (1-fpr))

# Find closest to 0.8
idx = np.nanargmin(np.abs(npv - 0.8))

# Find best gmean
idx2 = np.argmax(results.gmean)

# Display
display_npv_ppv_curve(ppv, npv, ths1, idx)

# Title
plt.suptitle("From 'roc_curve'")

# Show
print("\n\nResults from 'roc_curve'")
print(results)

"""
# ---------
# Option II
# ---------
# NOT WORKING!
#
# Compute the npv by knowing that it is the inverse
# of the precision, thus calling the function
# 'precision_recall_curve' with opposite labels and
# probabilities.

# .. note: invprec=npv
# .. note: invrec=fnr
# Computed inverted PR curve
invprec, invrec, invths2 = \
    precision_recall_curve(y_test, y_prob[:, 0],
        pos_label=clf.classes_[0])

# Create DataFrame
results = pd.DataFrame()
results['th'] = invths2[::-1]
results['npv'] = invprec[1:]
results['ppv'] = 0.0
results = results.sort_values(by='th')

# Find closest to 0.8
idx = np.nanargmin(np.abs(invprec - 0.8))

# Show
print("\n\nResults from 'precision_recall_curve'")
print(results)
print("\nIndex: {0} | Threshold: {1} | NPV: {2}" \
    .format(idx, invths2[idx-1], npv[idx]))

# Display graph
display_npv_ppv_curve(
    results.ppv,
    results.npv,
    results.th,
    idx)

# Title
plt.suptitle("From 'precision_recall_curve'")
"""

# ----------
# Option II
# ----------
# Perform the computation of metrics and the threshold
# search based on a condition (e.g. npv closest to an
# specific value) manually.
# Thresholds
thresholds = np.linspace(0,1,100)

# Metrics
def metrics(y_test, y_prob, th, **kwargs):
    # Libraries
    from sklearn.metrics import confusion_matrix
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_prob>th)
    tn, fp, fn, tp = cm.ravel()
    # Compute metrics
    return {'th': th,
            'ppv': tp/(tp+fp),
            'npv': tn/(tn+fn),
            'sens': tp/(tp+fn),
            'spec': tn/(tn+fp)}

# Compute scores
scores = [metrics(y_test, y_prob[:,1], t) \
    for t in thresholds]

# Create DataFrame
results = pd.DataFrame(scores)

# Find idx where npv is closest to 0.8
idx = np.nanargmin(np.abs(results.npv - 0.8))

# Show
print("\n\nResults from manual")
print(results)

# Display graph
display_npv_ppv_curve(
    results.ppv,
    results.npv,
    results.th,
    idx)

# Title
plt.suptitle("From 'manual thresholds'")

# Show
plt.show()