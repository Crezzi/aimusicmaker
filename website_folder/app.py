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


def process_midi_async(input_paths, session_id):
    """Process multiple MIDI files in background thread"""
    global processing_status
    try:
        print(f"Starting processing for session {session_id}")
        print(f"Processing {len(input_paths)} files: {input_paths}")
        processing_status[session_id] = {"status": "processing", "error": None}

        # Process each file or all files together - depends on your AI functions
        for i, input_path in enumerate(input_paths):
            print(f"Processing file {i+1}/{len(input_paths)}: {input_path}")
            # Update status to show progress
            processing_status[session_id] = {
                "status": "processing",
                "error": None,
                "progress": f"{i+1}/{len(input_paths)}",
            }

        # Your AI processing functions
        transpose()

        # Make sure tensor_AI outputs to the correct location
        tensor_AI(input_dir="website_folder/transposed", output_file="website_folder/download/new_ai_stuff.mid")

        # Mark processing as complete
        processing_status[session_id] = {"status": "complete", "error": None}
        print(f"Processing complete for session {session_id}")
    except Exception as e:
        print(f"Error processing MIDI files: {e}")
        processing_status[session_id] = {"status": "error", "error": str(e)}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        print(request.files)

        # Check if files were uploaded
        if "files" not in request.files:
            flash("No files selected")
            return redirect(request.url)

        files = request.files.getlist("files")  # Get list of files

        # Check if at least one file is selected
        if not files or all(file.filename == "" for file in files):
            flash("No files selected")
            return redirect(request.url)

        uploaded_files = []

        # Process each file
        for file in files:
            if file and file.filename != "":
                # Check if file is allowed (MIDI only)
                if not allowed_file(file.filename):
                    flash(f"File '{file.filename}' is not allowed. Only MIDI files (.mid, .midi) are accepted")
                    continue

                # Secure the filename
                if not file.filename:
                    quit()
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                # Save the uploaded file
                file.save(input_path)
                uploaded_files.append(input_path)
                print(f"Saved file: {filename}")

        # Check if we have any valid files
        if not uploaded_files:
            flash("No valid MIDI files were uploaded")
            return redirect(request.url)

        # Create unique session ID for this upload batch
        session_id = str(int(time.time())) + "_" + str(os.getpid())
        session["session_id"] = session_id
        session["output_filename"] = "new_ai_stuff.mid"  # Your AI outputs this filename
        session["uploaded_files"] = uploaded_files  # Store list of uploaded files

        # Initialize processing status
        global processing_status
        processing_status[session_id] = {"status": "starting", "error": None}

        # Start background processing (you may need to modify this function too)
        thread = threading.Thread(target=process_midi_async, args=(uploaded_files, session_id))
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


@app.route("/serve-midi/<filename>")
def serve_midi(filename):
    """Serve MIDI file for the HTML MIDI player (without forcing download)"""
    try:
        actual_filename = "new_ai_stuff.mid"
        full_file_path = os.path.abspath(os.path.join("website_folder", "download"))
        file_path = os.path.join("website_folder/download", actual_filename)

        print(f"Serving MIDI file: {file_path}")

        if not os.path.exists(file_path):
            print(f"MIDI file not found: {file_path}")
            return "File not found", 404

        # Serve file without forcing download (for the player)
        response = send_from_directory(
            directory=full_file_path,
            path=actual_filename,
            as_attachment=False,  # Don't force download
            mimetype="audio/midi",  # Set proper MIME type
        )

        # Add CORS headers if needed
        response.headers["Access-Control-Allow-Origin"] = "*"

        return response

    except Exception as e:
        print(f"Error serving MIDI file: {e}")
        return "Error serving file", 500


if __name__ == "__main__":
    app.run()
