from flask import Flask, render_template
app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/newpage")
def new_page():
    return render_template("newpage.html")

if __name__ == "__main__":
    app.run()