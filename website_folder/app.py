import os
from flask import Flask, render_template, request, redirect, url_for, flash
app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/newpage")
def new_page():
    return render_template("newpage.html")

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('files')  # Avoid naming conflict

    for file in uploaded_files:
        if file and file.filename:
            file.save(os.path.join('uploads', file.filename))

    return render_template('upload_page.html')
 
@app.route('/help')
def help_page():
    return render_template('help_page.html')

if __name__ == "__main__":
    app.run()