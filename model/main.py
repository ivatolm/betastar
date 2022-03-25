import logging

from .pipeline import pipeline

from configs.environ_cfg import *


def run():
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	pipeline()
