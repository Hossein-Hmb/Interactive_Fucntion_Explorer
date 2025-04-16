import os
import subprocess
from flask import Flask, send_from_directory, request, Response

app = Flask(__name__, static_folder='.', static_url_path='')

# Serve the landing page


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Proxy all requests under /app to the Streamlit server


@app.route('/app/<path:path>', methods=['GET', 'POST'])
def proxy(path):
    streamlit_url = f"http://localhost:8501/{path}"
    # Use curl to forward the request and grab the response
    result = subprocess.check_output(['curl', '-s', streamlit_url])
    return Response(result, mimetype='text/html')


if __name__ == '__main__':
    # Start Streamlit in the background
    subprocess.Popen([
        'streamlit', 'run', 'app.py', '--server.port', '8501'
    ])
    # Start Flask for Heroku
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
