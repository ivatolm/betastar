from pathlib import Path
import time

PLANS_DIR = str(Path().absolute()) + "/plans"
MEASUREMENT_DIR = str(Path().absolute()) + "/data/metrics/"
SAVES_DIR = str(Path().absolute()) + "/data/models"
VERSION = ''.join(str(time.time()).split('.'))
