from flask import Flask, request, render_template
#from markupsafe import escape
app = Flask(__name__)

@app.route("/")
def index():
    #celsius = str(escape(request.args.get("celsius", "") ))
    celsius = request.args.get("celsius", "")
    if celsius:
        fahrenheit = fahrenheit_from(celsius)
    else:
        fahrenheit = ""
    return render_template('index.html', fahrenheit=fahrenheit)

@app.route('/about/')
def about():
    return render_template('about.html')

def fahrenheit_from(celsius):
    """Convert Celsius to Fahrenheit degrees."""
    try:
        fahrenheit = float(celsius) * 9 / 5 + 32
        fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
        return str(fahrenheit)
    except ValueError:
        return "invalid input"




if __name__ == "__main__":
    #app.run(host="127.0.0.1", port=8080, debug=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
