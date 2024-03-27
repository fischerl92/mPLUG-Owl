import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# srun --pty -n 1 -c 2 --time=01:00:00 --mem=40G --gres=gpu:1  bash -l

import logging
logging.basicConfig(level=logging.INFO)

# Load via Huggingface Style
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch

import logging
logging.basicConfig(level=logging.INFO)

pretrained_ckpt = 'MAGAer13/mplug-owl-bloomz-7b-multilingual'

model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # move to cuda if available

logging.info("loaded model")

image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

logging.info("loaded everything else")

def translate(image_file, query):
    # We use a human/AI template to organize the context as a multi-turn conversation.
    # <image> denotes an image placehold.
    prompts = [query]
    image_list = [image_file]
    logging.info(f"Translating: {query} for image: {image_file}")

    # generate kwargs (the same in transformers) can be passed in the do_generate()
    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512
    }
    from PIL import Image
    images = [Image.open(_) for _ in image_list]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    logging.info(sentence)
    return sentence


if __name__ == "__main__":
    image_files = []
    for i in range(200):
        image_files.append(f"/net/cephfs/shares/iict-sp4.ebling.cl.uzh/ad-experiments/scripts/test_gpttranslationENDE/image_{i}.jpg")

    ads = open("../mPLUG-Owl2/source_ads.txt", "r").readlines()

    queries = [f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: <image>
    Human: Translate the following audio description for this image from English to German even if the audio description does not match the image: \n {ad.strip()}".
    AI: ''' for ad in ads]

    with open("translations.txt", "w") as outfile:
        for image_file, query in zip(image_files, queries):
            outfile.write(translate(image_file, query).replace("\n", " ").strip() + "\n")
