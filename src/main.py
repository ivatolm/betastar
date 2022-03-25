import logging
import argparse

from numpy import save

from .pipeline import pipeline

from configs.environ_cfg import *
from configs.data_cfg import *


def run():
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument("--load_version", dest="load_version", help="Load model from this version")
  parser.add_argument("--save_version", dest="save_version", help="Save model to this version")
  args = parser.parse_args()

  load_version = args.load_version
  save_version = args.save_version

  if load_version == "last" or save_version == "last":
    versions = set()
    if Path(SAVES_DIR).exists():
      for item in os.listdir(SAVES_DIR):
        version, _ = item.split('.')
        versions.add(int(version))
      last_version = str(max(versions))
      load_version = last_version if load_version == "last" else load_version
      save_version = last_version if save_version == "last" else save_version
    else:
      logging.info("run: no previous saves were found, so falling back to creating")
      load_version = None
      save_version = None

  pipeline_args = []
  if load_version is not None:
    pipeline_args.append(load_version)
  if save_version is not None:
    pipeline_args.append(save_version)
  pipeline(*pipeline_args)
