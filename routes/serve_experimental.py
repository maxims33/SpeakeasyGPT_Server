from flask_cors import cross_origin
from flask import request, jsonify
from api.schemas import format_response
from auth.auth import auth_required
#from replit.ai.modelfarm import CompletionModel
from api.schemas import ( 
  deserialize_request,
  format_response
)

from serve import app

@app.route("/ai", methods=['POST'])
@auth_required
@cross_origin()
def replitAIQuery():
  """ endpoint for setting the replit AI """
  try:
    req = deserialize_request(request)
    #model = CompletionModel("text-bison")
    #response = model.complete([req.prompt], temperature=0.2)
    
    #print(response.responses[0].choices[0].content)
    #return format_response(response.responses[0].choices[0].content)
    
    return format_response("Sample response " + req.prompt)
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')
