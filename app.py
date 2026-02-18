import os
import tempfile
import shutil
import uuid
import base64
import io
from flask import Flask, render_template, request, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from utils.segmentation import segment_image, extract_objects, create_zip
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Create a unique session ID for this upload
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'instance_seg_app', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_dir, filename)
            file.save(file_path)
            
            # API Token (optional, but recommended)
            api_token = os.getenv("HF_API_TOKEN") # User can set this in .env
            
            # 1. Segment
            try:
                segmentation_results = segment_image(file_path, api_token)
            except Exception as e:
                shutil.rmtree(session_dir)
                return jsonify({'error': f"Segmentation failed: {str(e)}"}), 500
            
            # 2. Extract Objects
            try:
                extracted_files = extract_objects(file_path, segmentation_results, session_dir)
            except Exception as e:
                 shutil.rmtree(session_dir)
                 return jsonify({'error': f"Extraction failed: {str(e)}"}), 500

            if not extracted_files:
                 shutil.rmtree(session_dir)
                 return jsonify({'error': "No objects detected."}), 200

            # 3. Create ZIP
            zip_filename = f"objects_{session_id}.zip"
            zip_path = os.path.join(session_dir, zip_filename)
            create_zip(extracted_files, zip_path)
            
            # Prepare response
            # We need to serve these files. 
            # Strategy: Return filenames and a session ID. 
            # The client can request /download/<session_id>/<filename>
            
            response_files = []
            for f in extracted_files:
                response_files.append(os.path.basename(f))
                
            return jsonify({
                'session_id': session_id,
                'files': response_files,
                'zip_file': zip_filename
            })
            
        except Exception as e:
            shutil.rmtree(session_dir)
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<session_id>/<filename>')
def download_file_route(session_id, filename):
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'instance_seg_app', session_id)
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404
        
    return send_file(file_path, as_attachment=True)

@app.route('/refine', methods=['POST'])
def refine_segmentation():
    """
    Refine an existing segmented image by applying an erasure mask.
    Expects JSON: {
        'session_id': str,
        'filename': str,
        'mask': str (base64 encoded image of the erasure strokes)
    }
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    session_id = data.get('session_id')
    filename = data.get('filename')
    mask_b64 = data.get('mask')
    
    if not all([session_id, filename, mask_b64]):
        return jsonify({'error': 'Missing parameters'}), 400
        
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'instance_seg_app', session_id)
    file_path = os.path.join(session_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        # Decode the mask
        if ',' in mask_b64:
            mask_b64 = mask_b64.split(',')[1]
        mask_bytes = base64.b64decode(mask_b64)
        erasure_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        # Load the original image
        original = Image.open(file_path).convert("RGBA")
        
        # Resize erasure mask to match image if needed
        if erasure_mask.size != original.size:
            erasure_mask = erasure_mask.resize(original.size, Image.NEAREST)
            
        # Apply the erasure
        # Where the mask is white (drawn), we want to make the image transparent.
        # We can do this by updating the alpha channel.
        
        r, g, b, a = original.split()
        
        # Invert the erasure mask (0=keep, 255=erase) -> we want 0=erase, 255=keep for composition?
        # Actually easier: new_alpha = old_alpha - erasure_value
        # If erasure is white (255), we want alpha to become 0.
        # If erasure is black (0), we want alpha to stay same.
        
        # Convert images to numpy for easy math
        import numpy as np
        a_arr = np.array(a)
        mask_arr = np.array(erasure_mask)
        
        # If mask is > 0, set alpha to 0. 
        # (Assuming the frontend sends a black canvas with white strokes for erasure)
        a_arr = np.where(mask_arr > 10, 0, a_arr)
        
        new_a = Image.fromarray(a_arr.astype('uint8'))
        original.putalpha(new_a)
        
        # Save back
        original.save(file_path, format="PNG")
        
        # Re-create ZIP to include updated file
        # Find all png files in the directory
        all_files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('.png') and 'mask' not in f]
        zip_filename = f"objects_{session_id}.zip"
        zip_path = os.path.join(session_dir, zip_filename)
        create_zip(all_files, zip_path)
        
        return jsonify({'status': 'success', 'filename': filename})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """
    Endpoint to manually trigger cleanup if needed, 
    though we might want to rely on a scheduled job or just aggressive cleanup.
    For this simple app, we'll let the client tell us when they are done, 
    OR we just leave it to OS temp cleaner (risk of filling up).
    Better: cleanup after ZIP download? No, user might want individual files.
    
    Let's stick to the user's requirement: "Automatically delete temp files after response completion."
    This is tricky with multiple download links. 
    A simple approach: cleaning up after X minutes or strictly after the single ZIP download if that was the only requirement.
    The user wants "Individual download buttons per object" AND "A single ZIP file".
    So we need to keep files for a bit.
    
    We won't implement auto-cleanup on *response* because the user needs to download *after* the response.
    We will rely on a strict user flow or a background thread?
    Background thread is best but complex for a single script.
    
    Simple Hack: Check for old folders in `instance_seg_app` on every upload request and delete those older than 10 mins.
    """
    cleanup_old_sessions()
    return jsonify({'status': 'cleaned'})

def cleanup_old_sessions():
    """Delete sessions older than 15 minutes"""
    base_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'instance_seg_app')
    if not os.path.exists(base_dir):
        return
        
    import time
    current_time = time.time()
    
    for session_id in os.listdir(base_dir):
        session_path = os.path.join(base_dir, session_id)
        if os.path.isdir(session_path):
             # check modification time
             if current_time - os.path.getmtime(session_path) > 900: # 15 mins
                 try:
                     shutil.rmtree(session_path)
                 except:
                     pass

# Run cleanup on startup
cleanup_old_sessions()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
