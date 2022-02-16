import streamlit as st
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


data_path = "/home/drew/DataScience/projects/streamlit/classification_intuition/dataset/heart_disease_health_indicators_BRFSS2015.csv"


@st.cache
def load_data(nrows):
    data = pd.read_csv(data_path, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis="columns", inplace=True)
    return data


st.title("Building Intuition for Classification")

""" __Goal__ :  Create a model that suggests whether or not one should be screened for heart disease."""
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
# st.sidebar.markdown(
#     " * Since we have such an unbalanced dataset we may want to oversample the minority class \
# (those with heart disease) in order to improve the model. Note that we don't touch the test set here. \
# "
# )
counts_i = Counter(y_train)
frac_i = np.around(counts_i[1] / (counts_i[0]), 2)
st.sidebar.write(
    "Ratio of people with heart disease to healthy people before oversampling: {} ".format(
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
    "Ratio of people with heart disease to healthy people after oversampling: {} ".format(
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
ns_probs = [0 for _ in range(len(y_test))]
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

accuracy_now = (tp+tn) / (tn + fp + fn + tp)
precision_now = (tp) / (tp+ fp)
recall_now = (tp) / (tp + fn)
tpr_now = tp / (tp + fn)
fpr_now = fp / (fp + tn)
fscore_now = 2*(precision_now*recall_now)/(precision_now+ recall_now)
precision_now,recall_now,fscore_now, _ = precision_recall_fscore_support(
        y_test, y_predicted, average="binary", zero_division=0
    )

performance_now_np = np.around(np.array([fscore_now, accuracy_now, precision_now, recall_now, tpr_now, fpr_now]),3)
performance_now = pd.DataFrame(columns=["fscore","accuracy", "precision", "recall", "tpr", "fpr"])
performance_now.loc[0] = performance_now_np

print(performance_now)
lam_list = []
lam_list.append(lam_final)  # Add the specific lambda chosen.
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


left_col, right_col = st.columns(2)
with left_col:

    # ROC AUC Plot
    st.subheader("ROC AUC Plot")

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    # plot the roc curve for the model
    plt.plot(fpr_now, tpr_now, 'rp', markersize=15, label = "chosen threshold")
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="no skill. AUC = {}".format(ns_auc))
    plt.plot(
        lr_fpr,
        lr_tpr,
        marker=".",
        label="LogReg. AUC = {}".format(np.around(lr_auc, 2)),
    )
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

    st.subheader("Decision Threshold")
    # Decision Threshold Plot.
    fig_dt, ax_dt = plt.subplots(figsize=(10, 8))

    ax_dt.plot(thresholds, accuracy, label="accuracy")
    ax_dt.plot(thresholds, precision, label="precision")
    ax_dt.plot(thresholds, recall, label="recall")
    ax_dt.plot(thresholds, fscore, label="fscore")
    ax_dt.axvline(x=threshold, color="b", label="DT implimented")
    ax_dt.legend(loc="upper left")
    ax_dt.set_xlabel("decision threshold")
    ax_dt.set_ylabel("score")
    st.pyplot(fig_dt)

    st.subheader("Regularization Paths")
    fig_rp, ax_rp = plt.subplots(figsize=(10, 6))

    def top_cols(dftemp, ncols):
        dfsum = dftemp.sum().to_frame().reset_index()
        dfsum = dfsum.sort_values(by=0, ascending=False, inplace=False).head(ncols)
        top_cols = dfsum["index"].tolist()
        return dftemp[top_cols]

    col_num = st.slider("Number of most influential features to show: ", 1, 10, value=5)
    sns.lineplot(data=top_cols(regularization_path, col_num))
    plt.xscale("log")
    plt.axvline(x=lam_final, color="b", label="lambda implimented")
    ax_rp.set_xlabel("lambda")
    ax_rp.set_ylabel("feature weight")
    ax_rp.legend()
    st.pyplot(fig_rp)

st.subheader("Model Summary")

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
        speed=0.3,
        reverse=False,
        loop=True,
        quality="low",  # medium ; high
        height=None,
        width=None,
        key=None,
    )


left, right = st.columns(2)
with left:

    st.subheader("Things to Consider")

    """
    * What if the CDC wanted to create a simple 3 question form to be used during routine check-ups \
    to determine who should be screened for heart disease. How would you decide which \
    three fearures to include? Why would l1 regularization be preferable for this task?
    * What is the most useful metric to determine the quality of a specific model? GO \
    iNTO ACCURACY, FSCORE,... 
    * We want to maximize recall, see here: https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
    * What is the effect of having such an imbalanced dataset? 
    """



with right:

    st.subheader("Definitions")

    """_Intuitive definitions of classification concepts._"""
    
    """__Correlation Matrix__: The correlation matrix describes relationships between variables. 
    A positive correlation means that both variables move in the same direction.
    A negative correlation means they move in opposite directions. """


st.subheader("References and Further Reading:")

""" 
* Get all plots to render as sns.
* Make clickable definitions for each concepts
* Sparsity: The CDC wants a model that only uses 3 features. 
* Confusion matrix is weird may need to be transposed.
* Talk about 3 question questionare idea for sparsity. 
* Have a things to think about section (sparsity) and a definitions section. 
* Add number of nonzero weights for model. 
* Take away general health as a parameter. 
* Add in a model report section with accuracy, precision, and stuff. 
maybe make a table. 
* Also add in references and further reading.
* Get markdown badge for github page. 
* Make code more organized and readable. 
 """
