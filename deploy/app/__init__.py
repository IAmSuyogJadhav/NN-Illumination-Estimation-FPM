from flask import Flask
from config import Config

app = Flask(__name__)
app.debug = True

app.config.from_object(Config)

from app import routes
# The bottom import is a workaround to circular imports,
