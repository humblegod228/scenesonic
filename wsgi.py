from streamlit.web import cli as stcli
import os
import subprocess

def app(environ, start_response):
    # Streamlit expects to be run in a subprocess, so we'll invoke it this way.
    streamlit_command = ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
    
    # Start Streamlit as a subprocess
    process = subprocess.Popen(streamlit_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Send the HTTP response headers
    status = '200 OK'
    response_headers = [('Content-type', 'text/html')]
    start_response(status, response_headers)

    # Wait for the process to finish and collect output
    stdout, stderr = process.communicate()

    # Combine stdout and stderr
    output = stdout + stderr

    return [output]

# This is the entry point for Gunicorn
if __name__ == "__main__":
    app(None, None)
