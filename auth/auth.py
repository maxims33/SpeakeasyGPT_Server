from auth.dummy import auth_required as dummy_auth
from auth.firebase import auth_required as firebase_auth
from env_params import env_config

""" Wrapper for auth implementation """
auth_required = dummy_auth if env_config["auth"] == "dummy" else firebase_auth
