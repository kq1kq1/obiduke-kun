import os
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pdf_processor import PDFProcessor

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key')

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', filename)
        file.save(pdf_path)
        
        processor = PDFProcessor()
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')
        
        try:
            processor.process_pdf(pdf_path, output_path)
            return send_file(output_path, as_attachment=True, download_name=f'processed_{filename}')
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

@app.route('/banners')
def banners():
    competitor_banners = os.listdir('uploads/banners/competitor')
    custom_banners = os.listdir('uploads/banners/custom')
    return render_template('banners.html', 
                         competitor_banners=competitor_banners,
                         custom_banners=custom_banners)

@app.route('/upload_banner', methods=['POST'])
def upload_banner():
    banner_type = request.form.get('banner_type', 'custom')
    
    if 'banner_file' not in request.files:
        flash('No file selected')
        return redirect(url_for('banners'))
    
    file = request.files['banner_file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('banners'))
    
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        banner_path = os.path.join(app.config['UPLOAD_FOLDER'], 'banners', banner_type, filename)
        file.save(banner_path)
        flash(f'Banner uploaded successfully: {filename}')
        return redirect(url_for('banners'))
    
    flash('Invalid file type. Please upload a PNG or JPG image.')
    return redirect(url_for('banners'))

@app.route('/delete_banner/<banner_type>/<filename>', methods=['POST'])
def delete_banner(banner_type, filename):
    if banner_type not in ['competitor', 'custom']:
        flash('Invalid banner type')
        return redirect(url_for('banners'))
    
    banner_path = os.path.join(app.config['UPLOAD_FOLDER'], 'banners', banner_type, secure_filename(filename))
    if os.path.exists(banner_path):
        os.remove(banner_path)
        flash(f'Banner deleted: {filename}')
    else:
        flash('Banner not found')
    
    return redirect(url_for('banners'))

if __name__ == '__main__':
    os.makedirs('uploads/pdfs', exist_ok=True)
    os.makedirs('uploads/banners/competitor', exist_ok=True)
    os.makedirs('uploads/banners/custom', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
