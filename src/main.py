import logging
import argparse
import json

from .pipeline import train_pipeline

from configs.environ_cfg import *
from configs.data_cfg import *


def run():
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument("--plan", dest="plan", help="Train model by this plan")
  parser.add_argument("--load_version", dest="load_version", help="Load model from this version")
  parser.add_argument("--save_version", dest="save_version", help="Save model to this version")
  parser.add_argument("--graphics", dest="graphics", help="Visualize environment")
  args = parser.parse_args()

  plan = args.plan
  load_version = args.load_version
  save_version = args.save_version
  graphics = args.graphics

  if plan is not None:
    load_status = False
    plan_path = PLANS_DIR + '/' + plan
    if Path(plan_path).exists():
      try:
        with open(plan_path, 'r') as file:
          plan = json.load(file)
        load_status = True
      except:
        logging.error("run: plan file corrupted")
    else:
      logging.error(f"run: plan file {plan_path} doesn't exist")
    if load_status == False:
      exit()
  else:
    logging.error(f"run: plan is not specified")
    exit()

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
      logging.warning("run: no previous saves were found, so falling back to creating")
      load_version = None
      save_version = None

  pipeline_args = {}
  if load_version is not None:
    pipeline_args["load_version"] = load_version
  if save_version is not None:
    pipeline_args["save_version"] = save_version
  if graphics is not None:
    pipeline_args["graphics"] = graphics
  train_pipeline(plan, **pipeline_args)
