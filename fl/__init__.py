# fl/__init__.py
from fl.model import get_model, FootprintClassifier
from fl.client import FLClient
from fl.server import FLServer, fedavg
from fl.gradient_clipping import clip_gradients
