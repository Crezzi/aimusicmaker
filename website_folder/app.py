import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from website_folder.midi_note_change import change_note_in_midi

app = Flask(__name__)
app.config["DEBUG"] = True

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/newpage")
def new_page():
    return render_template("newpage.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            output_filename = f'modified_{filename}'
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)

            file.save(input_path)

            # Change note 60 to 62 (customize as needed)
            change_note_in_midi(input_path, output_path, original_note=60, new_note=62)

            return redirect(url_for('download_file', filename=output_filename))

    return render_template('upload_page.html')

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

@app.route('/help')
def help_page():
    return render_template('help_page.html')

if __name__ == "__main__":
    app.run()
