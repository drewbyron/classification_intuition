import streamlit as st
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import requests

# ML
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

# Settings
plt.style.use("seaborn")
sns.set_context("paper", font_scale=2.5)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
ml_gif = load_lottieurl(
    "https://assets9.lottiefiles.com/private_files/lf30_8npirptd.json"
)


_left, _left, mid, _right, _right = st.columns(5)
with mid:

    st_lottie(
        ml_gif,
        speed=0.75,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=None,
        width=None,
        key=None,
    )


data_path = Path(__file__).parents[1] / 'classification_intuition/dataset/heart_disease_health_indicators_BRFSS2015.csv'
# data_path = "/home/drew/DataScience/projects/streamlit/classification_intuition/dataset/heart_disease_health_indicators_BRFSS2015.csv"


@st.cache
def load_data(nrows):
    data = pd.read_csv(data_path, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis="columns", inplace=True)
    return data


st.title("Building Intuition for Classification")
"""__Author__: Drew Byron. [email.](william.andrew.byron@gmail.com) [github.](http://github.com/drewbyron) [linkedin.](http://linkedin.com/in/drew-byron/) \
"""
""" __Goal__ :  You work for the CDC and you are tasked with creating a model that suggests whether or not one should be screened for heart disease."""
""" __Dataset__ : The data we will use to construct the model comes from the 2015 [Behavioral Risk 
Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_data.htm) Public health surveys, conducted 
by the CDC. The cleaned dataset and details on all features can be found 
[here](https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset).
 _Thanks_ to Alex Teboul for cleaning the data."""
""" __Desired Learning Outcome__ : Gain intuition for machine learning concepts by considering the real world problem of 
creating a machine learning model for suggesting whether or not one should be screened for heart disease based 
on lifestyle and health metrics. """
"""* * *"""
st.subheader("Use the sidebar to select model hyperparameters.")

data_load_state = st.text("Loading data...")
data = load_data(200000)
data_load_state.text("Data has loaded.")

if st.checkbox("Show raw data."):
    st.subheader("Raw data")
    st.write(data.head())

if st.checkbox("Show description of raw data."):
    st.subheader("Raw data description")
    st.write(data.describe())

if st.checkbox("Show correlation matrix."):

    st.subheader("Correlation Matrix")
    annot = False
    corr = data.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="Greens", annot=annot)
    ax_corr.legend()
    st.pyplot(fig_corr)

    st.write(data.describe())

# Begin by isolating the target variable.
all_X = data.drop(["HeartDiseaseorAttack", "GenHlth"], axis=1)
all_y = data["HeartDiseaseorAttack"]


st.sidebar.title("Specify Hyperparameters")
st.sidebar.write("We will use regularized logistic regression.")

st.sidebar.subheader("Select features to consider in analysis.")
st.sidebar.markdown("Or (by default) use all features.")

if st.sidebar.checkbox(
    "I would like to select specific features to consider in the analysis."
):
    columns = st.sidebar.multiselect(
        "What specifict features would you like to investigate?",
        all_X.columns,
        default=all_X.columns[0],
    )
    all_X = data[columns]


X_train, X_test, y_train, y_test = train_test_split(all_X, all_y)

# perform a robust scaler transform of the dataset
scaler = StandardScaler()

X_train_np = scaler.fit_transform(X_train.to_numpy())
X_train = pd.DataFrame(X_train_np, columns=all_X.columns)

X_test_np = scaler.transform(X_test.to_numpy())
X_test = pd.DataFrame(X_test_np, columns=all_X.columns)

st.sidebar.subheader("Select an oversampling ratio.")

counts_i = Counter(y_train)
frac_i = np.around(counts_i[1] / (counts_i[0]), 2)
st.sidebar.write(
    "Ratio of diseased people to healthy people in training set before oversampling: {} ".format(
        frac_i
    )
)

sample_coeff = st.sidebar.slider("over samping ratio: ", 0.15, 1.0, value=1.0)

oversample = RandomOverSampler(sampling_strategy=sample_coeff)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X_train, y_train)
counts_f = Counter(y_over)
frac_f = np.around(counts_f[1] / (counts_f[0]), 2)
st.sidebar.write(
    "Ratio of diseased people to healthy people in training set after oversampling: {} ".format(
        frac_f
    )
)


# """Remember that the goal is to decide if someone should be screened for heart
# disease or not. Not to maximize the f1 score of the model.... Explain here. """

st.sidebar.subheader("Choose a type of regularization.")
regularization_type = st.sidebar.radio("Regularization type", ("l1", "l2"))
st.sidebar.subheader("Select the amount of regularization.")

lam_final = st.sidebar.slider("lambda: ", 0.001, float(30000), value=100.0)

st.sidebar.subheader("Select a decision threshold.")
threshold = st.sidebar.slider("decision threshold: ", 0.01, 1.0, value=0.5)

### Fit the model and create plots to assess performance.

# Show the performance of the model:
lr = LogisticRegression(
    C=1 / lam_final, solver="liblinear", penalty=regularization_type
)

# Apply threshold to probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype("int")


# fit the model to the oversampled data.
lr.fit(X_over, y_over)
# predict probabilities
yhat = lr.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = np.arange(0.01, 1, 0.02)
thresholds = np.append(thresholds, [threshold])
thresholds.sort()
# evaluate each threshold

accuracy = [accuracy_score(y_test, to_labels(probs, t)) for t in thresholds]
bal_accuracy = [balanced_accuracy_score(y_test, to_labels(probs, t)) for t in thresholds]

precision = [
    precision_recall_fscore_support(
        y_test, to_labels(probs, t), average="binary", zero_division=0
    )[0]
    for t in thresholds
]
recall = [
    precision_recall_fscore_support(
        y_test, to_labels(probs, t), average="binary", zero_division=0
    )[1]
    for t in thresholds
]
fscore = [
    precision_recall_fscore_support(
        y_test, to_labels(probs, t), average="binary", zero_division=0
    )[2]
    for t in thresholds
]

# Find AUC and ROC curves
ns_probs = np.zeros(len(y_test))
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, probs)

# Get attributes of confusion matrix.
y_predicted = to_labels(probs, threshold)
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted, normalize = 'true').ravel()

# Summarize the performance of the specific threshold and lambda chosen. 
accuracy_now = accuracy_score(y_test, y_predicted, normalize = 'true')
bal_accuracy_now= balanced_accuracy_score(y_test, y_predicted)
# balanced_accuracy = (tp+tn) / (tn + fp + fn + tp)

tpr_now = tp / (tp + fn)
fpr_now = fp / (fp + tn)

# balanced_accuracy = (tpr_now+ (1.0- fpr_now))/2
# bal_acc=balanced_accuracy_score(y_test,y_predicted) # Somehow the same... 
precision_now,recall_now,fscore_now, _ = precision_recall_fscore_support(
        y_test, y_predicted, average="binary", zero_division=0
    )

performance_now_np = np.around(np.array([accuracy_now, bal_accuracy_now, precision_now, recall_now, fpr_now,fscore_now]),3)
performance_now = pd.DataFrame(columns=["accuracy", "balanced accuracy", "precision", "recall/tpr", "fpr","fscore"])
performance_now.loc[0] = performance_now_np

lam_list = []
lam_list.append(lam_final)  # Add the specific lambda chosen in the app.
condition = True
lam = 30000  # This forces all weights to zero.

while condition:
    lam_list.append(lam)
    lam = lam / 5
    condition = lam > 1


# Create an empty df to save regularization path.
regularization_path = pd.DataFrame(columns=X_train.columns)

i = 0
for lam in lam_list:

    lr = LogisticRegression(C=1 / lam, solver="liblinear", penalty=regularization_type)
    lr.fit(X_train, y_train)
    # y_predicted = lr.predict(X_test)
    regularization_path.loc[i] = lr.coef_[0]
    i += 1


regularization_path["lambda"] = lam_list
regularization_path = regularization_path.set_index("lambda", drop=True)

"""
***
## Model Report
"""
left_col, right_col = st.columns(2)
with left_col:

    # ROC AUC Plot
    st.subheader("ROC AUC Plot")

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    # plot the roc curve for the model
    plt.plot(fpr_now, tpr_now, 'rp', markersize=15, label = "chosen threshold")
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="no skill. AUC = {}".format(ns_auc), linewidth=2.5)
    plt.plot(
        lr_fpr,
        lr_tpr,
        marker=".",
        label="LogReg. AUC = {}".format(np.around(lr_auc, 2)),
        linewidth=2.5)
    # axis labels
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    # show the legend
    plt.legend()

    st.pyplot(fig_roc)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    if st.checkbox("Normalize confusion matrix."):
        norm = "true"
    else:
        norm = None

    # Confusion Matrix plot.
    cm = confusion_matrix(y_test, to_labels(probs, threshold), normalize=norm)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, annot_kws={"size": 16})  # font size
    ax_cm.set_xlabel("true label")
    ax_cm.set_ylabel("predicted label")
    st.pyplot(fig_cm)

with right_col:

    st.subheader("Decision Threshold Plot")
    # Decision Threshold Plot.
    fig_dt, ax_dt = plt.subplots(figsize=(10, 8))

    ax_dt.plot(thresholds, accuracy, label="accuracy", linewidth=2.5)
    ax_dt.plot(thresholds, bal_accuracy, label="balanced accuracy", linewidth=2.5)
    ax_dt.plot(thresholds, precision, label="precision", linewidth=2.5)
    ax_dt.plot(thresholds, recall, label="recall", linewidth=2.5)
    ax_dt.plot(thresholds, fscore, label="fscore", linewidth=2.5)
    ax_dt.axvline(x=threshold, color="b", label="chosen threshold")
    ax_dt.legend(loc="lower right",prop={'size': 19})
    ax_dt.set_xlabel("decision threshold")
    ax_dt.set_ylabel("score")
    st.pyplot(fig_dt)

    st.subheader("Regularization Paths Plot")
    fig_rp, ax_rp = plt.subplots(figsize=(10, 6))

    def top_cols(dftemp, ncols):
        dfsum = dftemp.sum().to_frame().reset_index()
        dfsum = dfsum.sort_values(by=0, ascending=False, inplace=False).head(ncols)
        top_cols = dfsum["index"].tolist()
        return dftemp[top_cols]

    col_num = st.slider("Number of most influential features to show: ", 1, 10, value=5)
    sns.lineplot(data=top_cols(regularization_path, col_num), linewidth=2.5)
    plt.xscale("log")
    plt.axvline(x=lam_final, color="b", label="chosen lambda")
    ax_rp.set_xlabel("lambda")
    ax_rp.set_ylabel("feature weight")
    ax_rp.legend()
    st.pyplot(fig_rp)

"""
### Model Summary
"""

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(performance_now)
# Number of non-zero weights:
nonzero_weights = np.count_nonzero(regularization_path.loc[lam_final])
st.write("Number of non-zero feature weights: {}".format(nonzero_weights))

if st.checkbox("Show feature weights."):
    st.subheader("Feature weights for chosen lambda ({})".format(lam_final))
    st.write(regularization_path.loc[lam_final])


"""***"""


# lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
def_gif = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_yAh844.json"
)


_left, _left, mid, _right, _right = st.columns(5)
with mid:
    st_lottie(
        def_gif,
        speed=0.6,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=None,
        width=None,
        key=None,
    )



"""
## Definitions
"""
left, right = st.columns(2)
with left:

    """
    ### Simple Metrics

    __True Positives/Negatives (TP/TN)__: Number of times the model correctly predicts the positive/negative class. _See Confusion Matix._

    __False Positives/Negatives (FP/FN)__: Number of times the model incorrectly predicts the positive/negative class. _See Confusion Matix._
    """
    """
    __True Positive Rate (TPR)__: _See ROC plot._
    """
    st.latex(r''' TPR = \frac{TP}{P} = \frac{TP}{TP + FN}  ''')
    """
    __False Positive Rate (FPR)__: _See ROC plot._
    """
    st.latex(r''' FPR = \frac{FP}{N} = \frac{FP}{FP + TN}  ''')

    """
    __Precision__: Probability a positive prediction was correct. _See Decision Threshold Plot._
    """
    st.latex(r''' Precision = \frac{TP }{TP + FP} ''')

    """
    __Recall__: Probability an actual positive was correctly identified. Identical to TPR. _See Decision Threshold Plot._
    """
    st.latex(r''' Recall = \frac{TP }{TP + FN} ''')

with right:

    """ 
    ### Advanced Metrics
    """
    """
    __Accuracy__: Fraction of correctly predicted labels out of all (both correct and incorrect) predictions. \
    Consider how poor a metric this can be for imbalanced datasets. A model that predicts that no one has heart disease \
    will have an accuracy of ~.9 in our dataset because most people in the set don't have heart disease. _See Decision Threshold Plot._
    """
    st.latex(r''' Accuracy = \frac{TP+TN}{TP+TN +FP +FN} ''')

    """
    __Balanced Accuracy__: Accuracy you would obtain if you had a balanced data set (normalized to size of P/N classes). This is generally a better metric for \
    assessing imbalanced datasets. A model that predicts that no one (or everyone) has heart disease \
    will have a score of .5 for balanced accuracy. _See Decision Threshold Plot._
    """
    st.latex(r''' Balanced \,\, Accuracy = \frac{TPR - FPR + 1}{2} ''')

    """
    __Fscore__: The harmonic mean of the precision and recall. Technically this is the F1-score. For many \
    classification tasks mazimizing the [fscore](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/) \
    is sensible because it indicates both good precision and good recall. _See Decision Threshold Plot._
    """
    st.latex(r''' Fscore = 2 \frac{Precision * Recall }{Precision + Recall} ''')

# lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
consider_gif = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_cqdzv4dr.json"
)


_left, _left, mid, _right, _right = st.columns(5)
with mid:
    st_lottie(
        consider_gif,
        speed=0.6,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=None,
        width=None,
        key=None,
    )

"""
## Things to Consider

* What if the CDC wanted to create a simple 3 question form to be used during routine check-ups \
to determine who should be screened for heart disease. How would you decide which \
three fearures to include on this form? Why would l1 regularization be preferable for this task? _Hint_: Look up why \
l1 regularization promotes [sparsity](https://blog.mlreview.com/l1-norm-regularization-and-sparsity-explained-for-dummies-5b0e4be3938a) \
(meaning it drives weights to zero). In addition to being easier to interpret, why would a sparse model be less computationally intense to train and implement?
* What is the most useful metric to determine the quality of a specific model? Is it accuracy (or even balanced accuracy)? \
Or does maximizing the recall (TPR) make more sense given that we want to be sure to screen all those who could have heart disease. Though of course \
we don't want to screen every single person; there is a balance to be struck. To truly strike this balance we would need to have some \
quantification of the cost of screening.
* What is the effect of having such an imbalanced dataset? Mess around with the model hyperparameters (in the sidebar) to investigate how the decision threshold and the oversampling \
ratio relate to one another. Consider why a decision threshold of .5 corresponds to the center of the ROC curve with an oversampling ratio of 1.
* Mess with the model hyperparameters to get a better sense for what the metrics definied above mean in practice. Try going to the extremes of the parameters (decision\
threshold = 0, or lambda = huge) to build intuition. 
* Now that you have some basic intuition for these concepts look into the resources below for a deeper understanding. Keep referring to this example \
problem as you read more. I personally find that unless I am working with a concrete example the heavy math makes my eyes glaze over! 
"""

"""
## References and Further Reading

* [Concise high level overview of classification in machine learning (Google's free ML course).](https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative)
* [Great discussion of classification on imbalanced datasets.](https://machinelearningmastery.com/what-is-imbalanced-classification/)
* [Visual explanation of precision and recall.](https://en.wikipedia.org/wiki/Precision_and_recall)
* [Metrics for imbalanced data sets.](https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba#:~:text=Recall%20and%20True%20Positive%20Rate,denominator%20contains%20the%20true%20negatives.)
* [Scikit-learn docs.](https://scikit-learn.org/stable/user_guide.html)
"""

