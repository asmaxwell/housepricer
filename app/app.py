from flask import Flask, request, render_template
from housepricer import modeler
#from markupsafe import escape
app = Flask(__name__)

@app.route("/")
def index():
    model = modeler.trainer("./data/", "./data/", "EvolveModel.save", None, False, "notrandom")
    model.is_model_fitted() #throw immediate exception if not fitted instead of at user input
    item_list = ["date", "postcode", "housetype", "newbuild", "estatetype", "transactiontype"]
    data_list = [request.args.get(item, "") for item in item_list]
    
    stringOut=""
    for item, data in zip(item_list, data_list):
        if data!="":
            stringOut+=f"{item}: {data}\n"
    if(data_list[0]!=""):
        XvalTest = model.X_test[1]
        model_pred = model.predict_values(XvalTest)
        stringOut+=f"Prediction: £{model_pred}"
    
    return render_template('index.html', stringOut=stringOut)

@app.route('/about/')
def about():
    return render_template('about.html')



if __name__ == "__main__":
    #app.run(host="127.0.0.1", port=8080, debug=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
