import os
import threading
import time

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from website_folder.midi_note_change import tensor_AI, transpose

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "your-secret-key"  # Needed for session to work

# Configuration for file uploads
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mid", "midi"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global dictionary to track processing status (instead of session)
processing_status = {}


def allowed_file(filename):
    """Check if file has allowed extension (MIDI files only)"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_midi_async(input_path, session_id):
    """Process MIDI file in background thread"""
    global processing_status
    try:
        print(f"Starting processing for session {session_id}")
        processing_status[session_id] = {"status": "processing", "error": None}

        # Your AI processing functions
        transpose()
        # Make sure tensor_AI outputs to the correct location
        tensor_AI(input_dir="website_folder/transposed", output_file="website_folder/download/new_ai_stuff.mid")

        # Mark processing as complete
        processing_status[session_id] = {"status": "complete", "error": None}
        print(f"Processing complete for session {session_id}")
    except Exception as e:
        print(f"Error processing MIDI: {e}")
        processing_status[session_id] = {"status": "error", "error": str(e)}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        print(request.files)

        # Handle both 'file' and 'files' field names
        if "file" in request.files:
            file = request.files["file"]
        elif "files" in request.files:
            file = request.files["files"]
        else:
            flash("No file selected")
            return redirect(request.url)

        # Check if file is selected
        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        # Check if file is allowed (MIDI only)
        if not allowed_file(file.filename):
            flash("Only MIDI files (.mid, .midi) are allowed")
            return redirect(request.url)

        if file:
            # Secure the filename
            if not file.filename:
                print("im awesome")
                quit()
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output_filename = f"modified_{filename}"

            # Save the uploaded file
            file.save(input_path)

            # Create unique session ID for this upload
            session_id = str(int(time.time())) + "_" + str(os.getpid())
            session["session_id"] = session_id
            session["output_filename"] = "new_ai_stuff.mid"  # Your AI outputs this filename

            # Initialize processing status
            global processing_status
            processing_status[session_id] = {"status": "starting", "error": None}

            # Start background processing
            thread = threading.Thread(target=process_midi_async, args=(input_path, session_id))
            thread.daemon = True
            thread.start()

            return redirect(url_for("loading"))

    return render_template("generate_new_music.html")


@app.route("/loading")
def loading():
    return render_template("loading.html")


@app.route("/result")
def result():
    output_filename = session.get("output_filename")
    session_id = session.get("session_id")

    # Check if processing is actually complete
    if not session_id:
        return redirect(url_for("upload"))

    global processing_status
    status_info = processing_status.get(session_id, {"status": "unknown", "error": None})

    if status_info["status"] != "complete":
        return redirect(url_for("loading"))

    if output_filename:
        return render_template("result.html", filename=output_filename)
    else:
        flash(message="Processing failed. Please try again.")
        return redirect(url_for("upload"))


@app.route("/download/<filename>")
def download_file(filename):
    """Download the processed MIDI file"""
    try:
        # The actual file is always "new_ai_stuff.mid" based on your tensor_AI function
        actual_filename = "new_ai_stuff.mid"
        full_file_path = os.path.abspath(os.path.join("website_folder", "download"))
        file_path = os.path.join("website_folder/download", actual_filename)

        print(f"Download route called with filename: {filename}")
        print(f"Trying to download file from: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

        if not os.path.exists(file_path):
            # List all files in the uploads directory for debugging
            upload_files = os.listdir("website_folder/download") if os.path.exists("website_folder/download") else []
            print(f"Files in uploads directory: {upload_files}")
            flash("File not found. Please process a MIDI file first.")
            return redirect(url_for("home"))

        print(f"Sending file: {file_path}")

        # Use send_from_directory correctly
        response = send_from_directory(
            directory=full_file_path,  # Directory path
            path=actual_filename,  # Filename within that directory
            as_attachment=True,
            download_name="AIMM.midi",  # Name shown to user
        )

        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        return response

    except Exception as e:
        print(f"Error in download: {e}")
        import traceback

        traceback.print_exc()
        flash(f"Error downloading file: {str(e)}")
        return redirect(url_for("home"))


@app.route("/help")
def help_page():
    return render_template("help_page.html")


if __name__ == "__main__":
    app.run()
