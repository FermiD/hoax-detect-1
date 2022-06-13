from django.apps import AppConfig
from django.conf import settings
from django.urls import path
from .views import HoaxPredict
import pickle5 as pickle

class ApiConfig(AppConfig):
    name = 'api'
    MODEL_FILE = os.path.join(settings.MODELS, "pac_model.pkl")
    model = pickle.load(MODEL_FILE)