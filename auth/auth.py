from env_params import env_config

""" Wrapper for auth implementation """

if env_config["auth"] == "dummy":
  from auth.dummy import auth_required as dummy_auth
  auth_required = dummy_auth
else:
  from auth.firebase import auth_required as firebase_auth
  auth_required = firebase_auth
