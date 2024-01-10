import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION_LLama2,
    CONV_VISION_Vicuna0,
    StoppingCriteriaSub,
)

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--cfg-path",
        type=str,
        help="path to configuration file.",
        default="eval_configs/tinygptv_stage1_2_3_eval.yaml",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


setup_seeds(42)

# ========================================
#             Model Initialization
# ========================================

conv_dict = {
    "pretrain_vicuna0": CONV_VISION_Vicuna0,
    "pretrain_llama2": CONV_VISION_LLama2,
}

print("Initializing Chat")
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [
    torch.tensor(ids).to(device="cuda:{}".format(args.gpu_id)) for ids in stop_words_ids
]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(
    model,
    vis_processor,
    device="cuda:{}".format(args.gpu_id),
    stopping_criteria=stopping_criteria,
)
print("Initialization Finished")


if __name__ == "__main__":
    img_path = "tests/test_files/1.png"
    img = Image.open(img_path)
    img = img.convert("RGB")

    chat_state = CONV_VISION.copy()

    img_list = []
    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)

    user_msg = "Please write a poem about the image"
    chat.ask(user_msg, chat_state)

    num_beams = 1
    temperature = 1.0
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    print(llm_message)
