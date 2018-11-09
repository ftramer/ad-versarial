from .yolo_parser import YoloParser
from logo_detection import CKPT_PATH
import sys

weights_path = sys.argv[1]
cfg_path = "logo_detection/cfg/yolo-adchoices.cfg"
out_path = CKPT_PATH

parser = YoloParser(cfg_path, weights_path, out_path)
parser.run()
