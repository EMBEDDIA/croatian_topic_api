import os
import json
# import wget

from flask import (
    Flask,
    jsonify,
    send_from_directory,
    request,
    redirect,
    url_for
)
from flask_sqlalchemy import SQLAlchemy
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_restx import Api, Resource, fields, abort, reqparse

from celery import Celery
import celery.states as states

from . import api_functions
from . import topic_model_classifier


# global variables
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND')
celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config.from_object("project.config.Config")
db = SQLAlchemy(app)
api = Api(app, version='1.0',
          title='UGC API services',
          description='REST APIs for processing user-generated content')
ns = api.namespace('comments_api', description='REST services API for news comments')


# input and output definitions

topic_model_single_input = api.model('TopicModelSingleInput', {
    'text': fields.String(required=True, description='input text for topic')
})
topic_model_single_output = api.model('TopicModelSingleOutput', {
    'suggested_label': fields.List(fields.String(), required=True, description='suggested label for topics'),
    'description': fields.List(fields.String(), required=True, description='description of suggested label'),
    'topic_words': fields.List(fields.String(), required=True, description='topic words')
 })

topic_model_list_input = api.model('TopicModelListInput', {
    'texts': fields.List(fields.String, required=True, description='input list of texts for topic')
})
topic_model_list_output = api.model('TopicModelListOutput', {
    'suggested_label': fields.List(fields.String(), required=True, description='suggested label for topics'),
    'description': fields.List(fields.String(), required=True, description='description of suggested label'),
    'topic_words': fields.List(fields.String(), required=True, description='topic words')
})

@ns.route('/topic_model/')
class TopicModelClassifier(Resource):
    @ns.doc('predict topic from single text')
    @ns.expect(topic_model_single_input, validate=True)
    @ns.marshal_with(topic_model_single_output)
    def post(self):
        topics = topic_model_classifier.predict([api.payload['text']])
        return {'suggested_label':topics['suggested_label'],
                'description':topics['description'],
                'topic_words':topics['topic_words'] }



@ns.route('/topic_model_list/')
class TopicModelListClassifier(Resource):
    @ns.doc('predict topic from list of texts')
    @ns.expect(topic_model_list_input, validate=True)
    @ns.marshal_with(topic_model_list_output)
    def post(self):
        topics = topic_model_classifier.predict(api.payload['texts'])
        return {'suggested_label': topics['suggested_label'],
                'description': topics['description'],
                'topic_words': topics['topic_words']}


@app.route("/health/")
#@app.doc('get information about the health of this API')
def health():
    return api_functions.health()

@app.route("/documentation/")
#@app.doc('get Swagger documentation about this API')
def documentation():
    return api_functions.documentation()

