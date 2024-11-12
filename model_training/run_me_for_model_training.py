from hg_data_collection import data_collection
from hg_model_training import model_training
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
data_collection()
model_training()