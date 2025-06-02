import os
import time

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from website_folder.midi_note_change import change_note_in_midi, transpose

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "your-secret-key"  # Needed for session to work

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print(request.files)
        if "file" in request.files:
            file = request.files["file"]
        else:
            file = request.files['files']
        if file:
            filename = file.filename
            input_path = os.path.join('uploads', filename)
            output_filename = f'modified_{filename}'
            output_path = os.path.join('website_folder/uploads', output_filename)

            file.save(input_path)
            session['output_filename'] = output_filename  # Save filename to use in /result

            # Start processing but delay final display for 30s
            #change_note_in_midi(input_path, output_path, original_note=60, new_note=62)
            transpose(input_path, output_path)

            return redirect(url_for('loading'))
    return render_template("generate_new_music.html")


@app.route("/loading")
def loading():
    return render_template("loading.html")  # This page will auto-redirect after 30s

@app.route("/result")
def result():
    output_filename = session.get('output_filename')
    if output_filename:
        return render_template("result.html", filename=output_filename)
    else:
        return redirect(url_for('upload'))  # fallback in case session fails

@app.route("/uploads/<filename>")
def download_file(filename):
    return send_from_directory("uploads", filename, as_attachment=True)

@app.route("/help")
def help_page():
    return render_template("help_page.html")

if __name__ == "__main__":
    app.run()
