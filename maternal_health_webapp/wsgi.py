# wsgi.py

from serverless_wsgi import handle_request
from api.index import app

def handler(event, context):
    return handle_request(app, event, context)

if __name__ == "__main__":
    app.run(debug=True)
