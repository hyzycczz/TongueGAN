import yaml
from yaml import Loader



with open('config.yml', 'r') as f:
    config = yaml.load(f, Loader=Loader)