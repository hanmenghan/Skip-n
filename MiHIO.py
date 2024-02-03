import os, random, argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining, AutoModelForCausalLM, InstructBlipForConditionalGeneration
import numpy as np
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
from transformers import LogitsProcessorList, LogitsProcessor


INSTRUCTION_TEMPLATE = {
    "llava-1.5-7b-hf": "USER: <image>\n<question>\nASSISTANT:",
    "llava-1.5-13b-hf": "USER: <image>\n<question>\nASSISTANT:",
    "bakLlava": "USER: <image>\n<question>\nASSISTANT:",
    "fuyu-8b": "<question>\n",
    "InstructBLIP-7b":"<question>\n",
}

MODEL_NAME = {
    "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b-hf": "llava-hf/llava-1.5-13b-hf",
    "bakLlava": "llava-hf/bakLlava-v1-hf",
    "fuyu-8b": "adept/fuyu-8b",
    "InstructBLIP-7b": "Salesforce/instructblip-vicuna-7b",
}

QUESTION_TEMPLATE = {
    "None": "Please describe this image in detail.",
    "MiHI":  "Please describe this image in detail in one paragraph.",
    "MiHO":  "Please describe this image in detail.",
    "MiHIO": "Please describe this image in detail in one paragraph.",
}

class MiHOProcessor(LogitsProcessor):
    def __init__(self, tokenizer, LineBreaks='\n\n\n\n', Lam=float("inf"), Mode='None'):
        self.tokenizer = tokenizer
        self.LineBreaksToken  = tokenizer(LineBreaks).input_ids[-1]
        self.Lam = Lam
        self.Mode = Mode
    def __call__(self, input_ids: torch.FloatTensor, scores: torch.FloatTensor):
        if self.Mode == 'MiHO' or self.Mode == 'MiHIO':
            scores[:, self.LineBreaksToken] = scores[:, self.LineBreaksToken]-self.Lam    
        return scores


def model_initialization(args):
    processor = AutoProcessor.from_pretrained(MODEL_NAME[args.model])
    if 'fuyu' in args.model:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME[args.model], torch_dtype=torch.float16).to(device)
    elif 'llava-1.5-7b-hf' in args.model:
        model = AutoModelForPreTraining.from_pretrained(MODEL_NAME[args.model], torch_dtype=torch.float16).to(device)
    elif 'llava-1.5-13b-hf' in args.model or 'bakLlava' in args.model:
        model = AutoModelForPreTraining.from_pretrained(
            MODEL_NAME[args.model],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True)
    elif 'InstructBLIP' in args.model:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            MODEL_NAME[args.model], 
            torch_dtype=torch.float16, 
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True            
            )
    model.eval()
    return model, processor


def setup_seeds(config):
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="Obtain the description information of the LLM to help understand the hallucination.")
parser.add_argument("--model", type=str, default='fuyu-8b')
parser.add_argument("--data_path", type=str, default="/Users/zongbo/Data/COCO_2014")
parser.add_argument("--save_path", type=str, default="./log/")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action='store_true', default=False)
parser.add_argument("--MiHIO", type=str, default='None', choices=['MiHI', 'MiHIO', 'MiHO', 'None'])
parser.add_argument("--Lam", type=float, default=float("inf"))
parser.add_argument("--experiments", type=str, default='greedy')
args = parser.parse_args()
setup_seeds(args)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


model, processor = model_initialization(args)
logitprocessor = MiHOProcessor(processor.tokenizer, Lam=args.Lam, Mode=args.MiHIO)

IMAGES_PATH = os.path.join(args.data_path, 'val2014')
ANNOTATIONS_PATH = os.path.join(args.data_path, 'annotations_trainval2014/annotations/instances_val2014.json')
img_files = os.listdir(IMAGES_PATH)
random.shuffle(img_files)
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_anns = json.load(f)

img_dict = {}
categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )

base_dir  = os.path.join(args.save_path, args.model, args.experiments)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

args_dict = vars(args)
args_json = json.dumps(args_dict, indent=4)
with open(os.path.join(base_dir, 'args.json'), 'w') as file:
    file.write(args_json)

for img_id in tqdm(range(len(img_files))):
    if img_id == 5000:
        break
    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id
    
    image_path  = os.path.join(IMAGES_PATH, img_file)
    raw_image = Image.open(image_path).convert("RGB")
    template = INSTRUCTION_TEMPLATE[args.model]
    question = QUESTION_TEMPLATE[args.MiHIO]
    
    prompt = template.replace("<question>", question)
    input = processor(text=prompt, images=raw_image, return_tensors="pt").to(device)
    input_length = input.input_ids.shape[1]
    if 'InstructBLIP' in args.model:
        input_length = 0
    with torch.inference_mode():
        with torch.no_grad():
            output = model.generate(**input, 
                                    max_length=1024, 
                                    do_sample=args.sample, 
                                    num_beams=args.beam,
                                    logits_processor=LogitsProcessorList([logitprocessor]),                
                                    )
            generated_token = output[:, input_length:]  
            generated_text = processor.batch_decode(generated_token, skip_special_tokens=True)[0]
            
    img_save["caption"] = generated_text

    with open(os.path.join(base_dir, 'MiHIO_{}-Lam_{}-beam_{}-s.jsonl'.format(args.MiHIO, args.Lam, args.beam, args.sample)), "a") as f:
        json.dump(img_save, f)
        f.write('\n')