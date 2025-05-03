import argparse
import os

from modules import settings

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
config_filename = os.path.join(script_path, 'config.json')

args: argparse.Namespace = None

opts: settings.Settings = None
