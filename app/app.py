from flask import Flask, request, render_template
from housepricer import modeler
from housepricer import data_wrangling as dw
from housepricer import better_postcodes as bpc
from datetime import datetime
app = Flask(__name__)

@app.route("/")
def index():
    #don't load data in future just use for testing purposes
    model = modeler.trainer(None, "./data/", "RandomModel.save", None, False, "notrandom")
    model.is_model_fitted() #throw immediate exception (during docker compose up) if not fitted instead of at user input
    item_list = ["date", "postcode", "housetype", "newbuild", "estatetype", "transactiontype"]
    data_list = {item : request.args.get(item, "") for item in item_list}
    
    #print(data_list)
    model_pred_string=prediction_string(data_list, model)
    #print(model_pred_string)    
    
    return render_template('index.html', stringOut=model_pred_string)

def prediction_string(data: dict, model : modeler.trainer) -> str:
    stringOut=""
    if (data["date"]!="") and (data["postcode"]!=""):
        parsed_data = parse_data(data)
        #print(parsed_data)
        model_pred = model.predict_values(parsed_data)
        #print(model_pred)
        stringOut=f"Prediction: £{model_pred[0]:,.0f}"
    return stringOut

def parse_data(data : dict) -> list:
    datestr = data["date"]
    date = datetime.strptime(datestr, '%Y-%m-%d')

    converter = bpc.better_postcodes('data/codepo_gb/Data/CSV/')
    coords = converter.query_postcodes([data["postcode"]])
    house_type = data["housetype"]
    new_build = data["newbuild"][0]
    estate_type = data["estatetype"][0]
    transaction_type = data["transactiontype"]

    pre_encoded =[ [date.year, date.month, date.day
            , coords["Eastings"].to_list()[0], coords["Northings"].to_list()[0]
            , house_type
            , new_build
            , estate_type
            , transaction_type]]
    wrangler = dw.wrangling(None, "data/", "Encoder.save")
    encoded_data = wrangler.apply_encoder(pre_encoded)[0]
    return encoded_data



@app.route('/about/')
def about():
    return render_template('about.html')



if __name__ == "__main__":
    #app.run(host="127.0.0.1", port=8080, debug=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
