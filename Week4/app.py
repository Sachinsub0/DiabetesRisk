from flask import Flask, render_template
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    sample = [X_test[0]]
    prediction = regressor.predict(sample)
    return render_template('index.html', prediction_text = "The patient's predicted diabetes risk is: " + str(prediction[0]))
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001, debug=True)
