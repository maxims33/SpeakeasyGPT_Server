"""
Firebase Token Authentication
"""
from flask import jsonify, request
from firebase_admin import auth, credentials, initialize_app
from datetime import datetime

# Initialize Firebase Admin
cred = credentials.Certificate(
    #TODO externalize key paths
    #"keys/maxims-firebase-firebase-adminsdk-hjau4-dca9d66150.json")
    "keys/bba-firebase-firebase-adminsdk-fbsvc-5b7bec3b6f.json")
initialize_app(cred)


def verify_firebase_token():
    """Verify Firebase ID token in Authorization header"""
    if not request.headers.get('Authorization'):
        return None
    try:
        token = request.headers['Authorization'].split(' ').pop()
        print(f"Authorization Token: {token}")
        log_token_to_file(token)
        decoded_token = auth.verify_id_token(token)
        #decoded_token = auth.verify_id_token(
        #    token, check_revoked=True
        #)  #Checks if revoked, though requires call to firebase
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None


def log_token_to_file(token):
    with open("token_log.txt", "a") as f:
        f.write(f"======== Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S} ======== \n")
        f.write(token)
        f.write("\n=================================================\n\n")


def auth_required(f):
    """Decorator to require authentication"""

    def decorated_function(*args, **kwargs):
        decoded_token = verify_firebase_token()
        if not decoded_token:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    decorated_function.__name__ = f.__name__
    return decorated_function
