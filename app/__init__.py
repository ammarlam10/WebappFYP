from flask import Flask
from config import Config
import sys
sys.path.append('/home/hishamsajid/Documents/FlaskFront/backend')

 
app = Flask(__name__)
app.config.from_object(Config)

from app import routes
