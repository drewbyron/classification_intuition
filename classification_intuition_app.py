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


data_path = "/home/drew/DataScience/projects/streamlit/heart_disease/dataset/heart_disease_health_indicators_BRFSS2015.csv"


@st.cache
def load_data(nrows):
    data = pd.read_csv(data_path, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis="columns", inplace=True)
    return data


st.title("Building Intuition for Classification")

""" __Goal__ :  Create a model that suggests whether or not one should be screened for heart disease."""
""" __Dataset__ : The data we will use to construct the model comes from the 2015 Behavioral Risk 
Factor Surveillance System (BRFSS) Public health surveys, conducted 
by the CDC. The dataset and details on all features can be found 
[here](https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset).
 _Thanks_ to Alex Teboul for cleaning the data."""
""" __Desired Learning Outcome__ : Gain intuition for machine learning concepts by considering the real world problem of creating a machine learning model 
for early disease screening. """
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

# Begin by isolating the target variable.
all_X = data.drop("HeartDiseaseorAttack", axis=1)
all_y = data["HeartDiseaseorAttack"]


st.sidebar.title("Specify a Model")
st.sidebar.write("We use regularized logistic regression for interpretability.")

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
regularization_type = st.sidebar.radio(
    "Regularization type", ("l1", "l2", "none")
)
st.sidebar.subheader("Select the amount of regularization.")
lambda_max = np.max(np.abs((2 * X_train.values.T.dot(y_train - np.average(y_train)))))

lam_final = st.sidebar.slider("lambda: ", 0.001, float(30000), value=100.0)

st.sidebar.subheader("Select a decision threshold.")
threshold = st.sidebar.slider("decision threshold: ", 0.01, 1.0, value=0.5)

### Fit the model and create plots to assess performance.

# Show the performance of the model:
lr = LogisticRegression(
    C=1 / lam_final, solver="liblinear", penalty=regularization_type
)

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype("int")


# fit the model
lr.fit(X_over, y_over)
# predict probabilities
yhat = lr.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = np.arange(0.01, 1, 0.05)
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

lam_list = []
accuracy_list = []

# Create an empty df to save regularization path.
regularization_path = pd.DataFrame(columns=X_train.columns)
#     confusion_matrixs = pd.DataFrame(columns=["tn", "fp", "fn", "tp"])

lam = 30000
condition = True
i = 0
while condition:

    lam_list.append(lam)
    lr = LogisticRegression(C=1 / lam, solver="liblinear", penalty=regularization_type)

    lr.fit(X_train, y_train)

    y_predicted = lr.predict(X_test)

    regularization_path.loc[i] = lr.coef_[0]
    #         tn, fp, fn, tp = confusion_matrix(y_test, y_predicted, normalize = True).ravel()

    lam = lam / 5
    condition = lam > 1
    i += 1


regularization_path["lambda"] = lam_list
regularization_path = regularization_path.set_index("lambda", drop=True)


# st.pyplot(fig2)


left_col, right_col = st.columns(2)
with left_col:

    st.subheader("Correlation Matrix")

    annot = False

    corr = all_X.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, cmap="Greens", annot=annot)
    ax_corr.legend()
    st.pyplot(fig_corr)

    st.subheader("Confusion Matrix")

    if st.checkbox("Normalize confusion matrix."):
        norm = "true"
    else:
        norm = None

    # Confusion Matrix plot.
    cm = confusion_matrix(y_test, to_labels(probs, threshold), normalize=norm)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 5))
    # sns.set(font_scale=1.4)  # for label size
    sns.heatmap(cm, annot=True, annot_kws={"size": 16})  # font size
    ax_cm.set_xlabel("True")
    ax_cm.set_ylabel("Predicted")
    st.pyplot(fig_cm)

with right_col:

    st.subheader("Decision Threshold")
    # Decision Threshold Plot.
    fig_dt, ax_dt = plt.subplots(figsize=(10, 6))

    ax_dt.plot(thresholds, accuracy, label="accuracy")
    ax_dt.plot(thresholds, precision, label="precision")
    ax_dt.plot(thresholds, recall, label="recall")
    ax_dt.plot(thresholds, fscore, label="fscore")
    ax_dt.axvline(x=threshold, color="b", label="DT implimented")
    ax_dt.legend()
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


    # st.write("Select some of the most influential features.")
    col_num = st.slider("Number of most influential features to show: ", 1, 10, value=5)
    sns.lineplot(data=top_cols(regularization_path, col_num))
    plt.xscale("log")
    plt.axvline(x=lam_final, color="b", label="lambda implimented")
    ax_rp.set_xlabel("lambda")
    ax_rp.set_ylabel("feature weight")
    ax_rp.legend()
    st.pyplot(fig_rp)

