from flask import Flask, request, render_template
import pandas as pd
import numpy as np

import joblib

from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import NMF

import model



app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def data():
    if request.method == "POST":
        f = request.form['csvfile']
        with open(f) as file:
            test_data = pd.read_csv(file)

        # extracting the important features from using the train dataset
        data = pd.read_csv("train.csv")
        col_init, col_final, features, label = model.preprocessing(data)

        # removing the ID column
        # separating the features with ID column
        features_test = test_data.drop(['ID'], axis=1)

        # applying the log-transformation
        features_test = np.log1p(features_test)

        # removing the constant features
        cols_with_one_value = features_test.columns[features_test.nunique() == 1]
        features_test.drop(cols_with_one_value, axis=1, inplace=True)

        # feature decomposition using PCA
        pca = PCA(n_components=20)
        features_test_PCA = pca.fit_transform(features_test)

        # feature decomposition using SRP
        srp = SparseRandomProjection(n_components=75, eps=0.28, dense_output=False)
        features_test_SRP = srp.fit_transform(features_test)

        # feature decomposition using NMF
        nmf = NMF(n_components=40, init=None, solver="cd", beta_loss="frobenius",
                  tol=0.0001, max_iter=200, random_state=None, alpha=0.0,
                  l1_ratio=0.0, verbose=0, shuffle=False)
        features_test_NMF = nmf.fit_transform(features_test)

        # concatenating the decomposed features all together
        features_test_PCA = features_test_PCA[:, :20]
        features_test_SRP = features_test_SRP[:, :75]
        features_test_NMF = features_test_NMF[:, :40]
        features_test_dec = np.hstack((features_test_SRP, features_test_PCA, features_test_NMF))

        # feature engineered for better results
        eng_features_test_ = features_test.iloc[:, :].progress_apply(model.aggregate_row, axis=1)

        ## concatinating the new 10 features
        features_test = pd.concat([features_test, eng_features_test_], axis=1)

        # concatenating the data with decomposition components
        features_test_dec = pd.DataFrame(features_test_dec)
        features_test = pd.concat([features_test, features_test_dec], axis=1)

        # finding the feature importance using the same model and returning top 150 features
        features_test = features_test[col_final]

        # load the best model
        best_model = joblib.load("best_model.pkl")

        y_hat = best_model.predict(features_test)

        y_pred = np.expm1(y_hat)

        # making data more presentable
        pd.set_option('display.float_format', '{:.2f}'.format)

        pred_ = pd.DataFrame({'ID': test_data["ID"], 'Prediction': y_pred})

        pred_table = pred_.to_html()

        return pred_table

if __name__ == "__main__":
    app.run(debug=True)









