import os
import yaml
import gdown
from ast import literal_eval
from pydantic import BaseModel

try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources


def load_config(yaml_file: str):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file)

    return config


class Config(dict):
    def __init__(self, config):
        super(Config, self).__init__(**config)
        self.__dict__ = self

    @staticmethod
    def load_config():
        """
        Load config from config.yml file in face_recognition package
        Returns: Dict
        """
        with resources.open_text('api', 'config.yml') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return Config(cfg)


def load_classes():
    with resources.open_text('api', 'classes.txt') as textfile:
        classes = literal_eval(textfile.read())

    return classes


def download_weights(url, cache=None, md5=None, quiet=False):

    return os.path.join(gdown.cached_download(url, path=cache, md5=md5, quiet=quiet))


class ImageInput(BaseModel):
    base64_img: str
