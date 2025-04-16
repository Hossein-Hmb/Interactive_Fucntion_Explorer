import os
import subprocess
import requests
from flask import Flask, send_from_directory, request, Response, redirect

app = Flask(__name__, static_folder='.', static_url_path='')

# Serve the landing page


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/app')
def _app_root():  # redirect to have trailing slash
    return redirect('/app/')

# Proxy all requests under /app to the Streamlit server


@app.route('/app/', methods=['GET', 'POST'])  # serve the root of Streamlit
@app.route('/app/<path:path>', methods=['GET', 'POST'])
def proxy(path=''):
    # Build the target URL
    target_url = f"http://localhost:8501/{path}"
    # Forward headers (except Host) and cookies
    headers = {key: value for key,
               value in request.headers if key.lower() != 'host'}
    # Make the request to Streamlit
    resp = requests.request(
        method=request.method,
        url=target_url,
        headers=headers,
        params=request.args,
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        stream=True
    )
    # Build a Flask response
    excluded_headers = ['content-encoding',
                        'content-length', 'transfer-encoding', 'connection']
    response_headers = [(name, value) for name, value in resp.raw.headers.items(
    ) if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, response_headers)


if __name__ == '__main__':
    # Start Streamlit in the background
    subprocess.Popen([
        'streamlit', 'run', 'app.py', '--server.port', '8501'
    ])
    # Start Flask for Heroku
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
