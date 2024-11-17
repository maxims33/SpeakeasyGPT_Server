from marshmallow import Schema, fields, post_load
from marshmallow_enum import EnumField
from speakeasy.llmfactory import LLMType

# ------- Serializing / Deserializing ------------------

class Request():
  """ Request object and handlers """

  def __init__(self, prompt, llm_type=None):
    self.prompt = prompt
    self.llm_type = llm_type  # if None should then use whichever factory instantiated at start


class RequestSchema(Schema):
  """ Request schema """
  prompt = fields.Str(required=True)
  llm_type = EnumField(LLMType,
                       required=False,
                       missing=None,
                       by_value=True,
                       allow_none=True)  # Not sure exactly which did the trick

  @post_load()
  def make_request(self, data, **kwargs):  #pylint: disable=unused-argument
    """ Instantiate the Request object """
    return Request(**data)


def deserialize_request(req_json):
  """ Deserielze the request json """
  return RequestSchema(partial=True).load(req_json.get_json())


# Response class and hanlders
class Response():
  """ Response object """

  def __init__(self, resp, image=None):
    self.response = resp
    if image is not None:
      self.image = image


class ResponseSchema(Schema):
  """ The response schema"""
  response = fields.Str()
  image = fields.Str()


def format_response(respstr):
  """ format the response json """
  schema = ResponseSchema(many=False, partial=True)
  respobj = Response(respstr)

  # Basic image handling
  image_tag = 'BASE64ENCODED:'
  if respstr.startswith(image_tag):
    respobj.response = 'Here is the image.'
    respobj.image = respstr[len(image_tag):]  # Strip BASE64ENCODED:

  return schema.dump(respobj)

# AccountSettings

class AccountSettings():
  """ Request object and handlers """

  def __init__(self,
               id=None,
               username=None,
               password=None,
               email=None,
               fullname=None,
               gender=None,
               orientation=None,
               dob=None):
    self.id = id
    self.username = username
    self.password = password
    self.email = email
    self.fullname = fullname
    self.gender = gender
    self.orientation = orientation
    self.dob = dob


class AccountSettingsSchema(Schema):
  """ Request schema """
  #  id = fields.Int(required=False)
  username = fields.Str(required=False)
  password = fields.Str(required=False)
  fullname = fields.Str(required=False)
  gender = fields.Str(required=False, allow_none=True)
  orientation = fields.Str(required=False, allow_none=True)
  dob = fields.Str(required=False, allow_none=True)
  email = fields.Str(required=False, allow_none=True)

  @post_load()
  def make_request(self, data, **kwargs):  #pylint: disable=unused-argument
    """ Instantiate the Request object """
    return AccountSettings(**data)


def deserialize_AccountSettings(req_json):
  """ Deserielze the request json """
  return AccountSettingsSchema(partial=True).load(req_json.get_json())


# Handle id
# Handle password
# Is this class even used?
class AccountSettingsResponseSchema(Schema):
  """ The response schema"""
  #  id = fields.Int(required=False)
  username = fields.Str(required=False)
  password = fields.Str(required=False)  # To remove
  fullname = fields.Str(required=False)
  gender = fields.Str(required=False)
  orientation = fields.Str(required=False)
  dob = fields.Str(required=False)
  email = fields.Str(required=False)

def format_AccountSettings(respobj):
  """ format the response json """
  schema = AccountSettingsSchema(many=False, partial=True)
  return schema.dump(respobj)
