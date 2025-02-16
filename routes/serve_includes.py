from flask_cors import cross_origin
from flask import jsonify, request
from api.schemas import format_response
from speakeasy.nutrition_content import retrieve_menu
from auth.auth import auth_required

# Below works, but using Blueprints or managing the imports using packages and __init__.py is considered a better approach
from serve import app, env_config

@app.route("/nutrition_content", methods=['GET'])
@auth_required
@cross_origin()
def nutrition_content():
  """ endpoint for retriving the nutrition content """
  try:
    category = request.args.get('category')
    return jsonify(retrieve_menu(
      category=category,       
      image_folder=env_config['image_directory']))
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')
