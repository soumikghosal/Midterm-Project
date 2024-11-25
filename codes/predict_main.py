import pickle

from flask import Flask
import xgboost as xgb
from flask import request
from flask import jsonify

with open("../models/dv.bin", "rb") as file_in:
    dv = pickle.load(file_in)
    
with open("../models/lr.bin", "rb") as file_in:
    lr = pickle.load(file_in)

with open("../models/rf.bin", "rb") as file_in:
    rf = pickle.load(file_in)

with open("../models/xgb.bin", "rb") as file_in:
    xgb_model = pickle.load(file_in)

    
app = Flask("Diabetes")

@app.route("/")
def hello_world():
    return "<p>HWorld!</p>"

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    
    # DictVectorizer
    X_test = dv.transform([data])    
    
    # Logistic Regression
    lr_test_pred = lr.predict(X_test)[0]
    # Random Forest
    rf_test_pred = rf.predict(X_test)[0]
    # XGBoost
    features = dv.feature_names_
    dtest = xgb.DMatrix(X_test[0].reshape(1, -1), feature_names=features)
    y_pred = round(xgb_model.predict(dtest)[0])   
    
    # Ensembled
    lst = [lr_test_pred, rf_test_pred, y_pred]
    final_pred = max(set(lst), key=lst.count)    
    
    result = {
        'diabetes_check': bool(final_pred)
    }
    
    return jsonify(result)
    
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)