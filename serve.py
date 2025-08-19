"""
Flask rest service implementation
"""
import langchain
from langchain_core.prompts import PromptTemplate
import flask
import json
import base64
from  google.cloud import speech, texttospeech
from flask import Flask, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS, cross_origin
from langchain.chains import RetrievalQA
from sqlalchemy.orm import Session

from env_params import env_config
from speakeasy.orm.models import (
  User,
  engine,
  find_user,
  find_user_with_password,
  insert,
  update,
)
from speakeasy.customagents import init_agent, init_conversational_agent
from speakeasy.indexes import load_document_db, load_image_db
from speakeasy.llmfactory import LLMType, init_factory, init_factory_from_type
from speakeasy.quiz_content import generate_quiz_questions
from speakeasy.nutrition_content import generate_menu
from api.schemas import ( 
  deserialize_request,
  deserialize_AccountSettings,
  format_AccountSettings,
  format_response
)
from auth.auth import auth_required

# Flask app
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Set langchain's debug flag
langchain.debug = env_config['debug']

# Initialise the factory
factory = init_factory(env_config)

# Setup the VectorStore DB
doc_db = load_document_db(factory,
                          persist_dir=env_config['docs_persist_directory'])
img_db = load_image_db(factory,
                       persist_dir=env_config['images_persist_directory'])

# Setup the Agents
agent = init_agent(factory, doc_db, img_db, max_iterations=1)
convo_agent = init_conversational_agent(factory, doc_db, img_db)

# Setup factory dict with defaults
# Retrieve from a dict if already instansitated
factory_dict = {
    'LLMType.GOOGLE': None,
    'LLMType.GOOGLEAISTUDIO': None,
    'LLMType.HUGGINGFACE': None,
    'LLMType.LOCAL': None,
    'LLMType.OPENAI': None
}
factory_dict[str(LLMType(env_config['factory_type']))] = factory

def choose_factory(req):
  """ Not applicable for agents since these were initialised at start up """
  if req.llm_type is None:
    return factory
  if factory_dict[str(req.llm_type)] is not None:
    return factory_dict[str(req.llm_type)]

  # Instansiate new factory since not found in dict
  factory_dict[str(req.llm_type)] = init_factory_from_type(
      req.llm_type, env_config)
  return factory_dict[str(req.llm_type)]


# --------- Define Routes ------------------------------

# Import additional routes
import routes.serve_includes

@app.route("/")
def ping():
  """ endpoint to verify services are alive """
  return format_response("Ping.")

# Filename to be validated
@app.route('/images/<filename>')
def server_image(filename):
    return send_from_directory(env_config['image_directory'], filename)
  
@app.route("/query", methods=['POST'])
@auth_required
def query_llm():
  """ endpoint for basic LLM prompt """
  try:
    req = deserialize_request(request)
    fact = choose_factory(req)
    print(f"Factory: {fact}")
    return format_response(fact.llm.invoke(req.prompt))
    #return format_response(fact.llm.invoke(req.prompt).content) # ChatGoogleGenerativeAI
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/query_streaming", methods=['POST'])
@auth_required
#@stream_with_context
def query_streaming():
  """ endpoint for basic LLM prompt """

  #def stream_output(text):
    #yield f"data: {text}".encode('utf-8') + b'\n\n'  # Encode as bytes

  req = deserialize_request(request)
  fact = choose_factory(req)

  prompt = PromptTemplate(input_variables=["input"], template="{input}")
  # Create a Streaming Chain
  # Unclear yet how to do streaming without using expression language
  chain = prompt | fact.llm 
  
  response = flask.Response(chain.stream({'input': req.prompt}))
  response.content_type = "text/event-stream"
  return response


@app.route("/search_docs", methods=['POST'])
@auth_required
def search_docs():
  """ endpoint for querying the document vectorstore """
  try:
    req = deserialize_request(request)
    fact = choose_factory(req)
    chain = RetrievalQA.from_chain_type(
        llm=fact.llm,
        chain_type="stuff",
        retriever=doc_db.as_retriever(search_kwargs={"k": factory.max_k}),
        input_key="question",
        return_source_documents=True)
    return format_response(chain.invoke(req.prompt)['result'])
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/search_images", methods=['POST'])
@auth_required
def search_images():
  """ endpoint for querying the image vectorstore """
  try:
    req = deserialize_request(request)
    fact = choose_factory(req)
    chain = RetrievalQA.from_chain_type(
        llm=fact.llm,
        chain_type="stuff",
        retriever=img_db.as_retriever(search_kwargs={"k": factory.max_k}),
        input_key="question",
        return_source_documents=True)
    return format_response(chain.invoke(req.prompt)['result'])
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/run_agent", methods=['POST'])
@auth_required
def run_agent():
  """ endpoint for zero / few shot agent """
  try:
    req = deserialize_request(request)
    return format_response(agent.invoke(req.prompt)['output'])
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/run_convo_agent", methods=['POST'])
@auth_required
def run_convo_agent():
  """ endpoint for conversational agent"""
  try:
    req = deserialize_request(request)
    return format_response(convo_agent.invoke(input=req.prompt)['output'])
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/authUser", methods=['POST'])
@cross_origin()
def authenticateUser():
  """ endpoint for basic user authentication """
  try:
    req = deserialize_AccountSettings(request)
    session = Session(engine)
    respobj = find_user_with_password(req.username, req.password, session)
    if respobj is None:
      return '{"error": "authentication failed"}'
    return format_AccountSettings(respobj)
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/accountSettings", methods=['GET'])
@auth_required
@cross_origin()
def getAccountSettings():
  """ endpoint for getting account settings """
  try:
    req_user = request.args.get('username')
    session = Session(engine)
    respobj = find_user(req_user, session)
    if respobj is None:
      return "{}"
    return format_AccountSettings(respobj)
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


@app.route("/accountSettings", methods=['POST'])
@auth_required
@cross_origin()
def setAccountSettings():
  """ endpoint for setting the account settings """
  try:
    req = deserialize_AccountSettings(request)
    session = Session(engine)
    u = find_user(req.username, session)
    if u is None:
      u = User(username=req.username,
               password=req.password,
               fullname=req.fullname,
               dateOfBirth=req.dob,
               orientation=req.orientation,
               gender=req.gender)
      insert(u, session)
    else:
      if req.username: u.username = req.username
      if req.password: u.password = req.password
      if req.fullname: u.fullname = req.fullname
      if req.dob: u.dateOfBirth = req.dob
      if req.orientation: u.orientation = req.orientation
      if req.gender: u.gender = req.gender
      update(u, session)
    return "{}"
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')

@app.route("/generate_quiz_content", methods=['GET'])
@auth_required
@cross_origin()
def generate_quiz_content():
  """ endpoint for retriving the quiz questions """
  try:
    category = request.args.get('category')
    return generate_quiz_questions(factory.llm, category).json()
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')

@app.route("/generate_nutrition_content", methods=['POST'])
@auth_required
@cross_origin()
def generate_nutrition_content():
  """ endpoint for retriving the nutrition questions """
  try:
    req = deserialize_request(request)
    fact = choose_factory(req)
    return generate_menu(fact.llm, req.prompt).json()
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')

@app.route("/speech_to_text", methods=['POST']) 
@auth_required
@cross_origin()
def speech_to_text():
  """ endpoint for converting speech to text """
  try:
    #req = json.loads(request.data)
    #print(f"Speech to Text request data: {request.get_json()}")

    audio = speech.RecognitionAudio(content=base64.b64decode(request.get_json()['audioData']))

    config = speech.RecognitionConfig(
       # Some of the supported encoding options are: 
       # MP3, LINEAR16, AMR (encoding for video/3gpp ?), amongst others
       # See https://cloud.google.com/speech-to-text/docs/encoding
       encoding=speech.RecognitionConfig.AudioEncoding.MP3,
       sample_rate_hertz=8000, # Must be 8000 for AMR. Comment out for LINEAR16
       language_code='en-US'  # Replace with your desired language code
    )

    # Detects speech in the audio file
    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)

    first_alternative = ""
    for result in response.results:
       # First alternative is the most probable result
       first_alternative = first_alternative + result.alternatives[0].transcript
    print(u'Transcript: {}'.format(first_alternative))
    return json.dumps({ 'text': first_alternative })
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')

@app.route("/text_to_speech", methods=['POST']) 
@auth_required
@cross_origin()
def text_to_speech():
  """ endpoint for converting speech to text """
  try:
    llm_response = factory.llm.invoke(request.get_json()['text'])
    
    client = texttospeech.TextToSpeechClient()

    # Set the voice configuration
    voice = texttospeech.VoiceSelectionParams(
       language_code="en-US",
       name="en-US-Wavenet-C"
    )

    # Set the audio configuration
    audio_config = texttospeech.AudioConfig(
       # Some of the supported encoding options are: MP3, LINEAR16
       # Refer https://cloud.google.com/python/docs/reference/texttospeech/latest/google.cloud.texttospeech_v1.types.AudioEncoding.html
       audio_encoding=texttospeech.AudioEncoding.MP3
    )

    #Perform the text-to-speech synthesis
    response = client.synthesize_speech(
       request={"input": texttospeech.SynthesisInput(text=llm_response), "voice": voice, "audio_config": audio_config}
    )

    print("Sending response")
    return json.dumps({
      'audio': base64.b64encode(response.audio_content).decode('utf-8')
    })
  except Exception as exc:
    print(f"Caught exception: {exc}")
    return format_response('Oops sorry an error occured.')


#------------- For debugging  -----------------------------

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
