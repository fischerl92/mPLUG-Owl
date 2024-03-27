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

def translate(image_list, query):
    # We use a human/AI template to organize the context as a multi-turn conversation.
    # <image> denotes an image placehold.
    prompts = [query]
    logging.info(f"Translating: {query} for image: {image_list}")

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
    image_files = ["../../image_198_0.jpg", "../../image_198_50.jpg", "../../image_198_100.jpg", "../../image_198_150.jpg", "../../image_198_200.jpg"]

    query = f'''The following is a conversation between a curious human and AI assistant. The assistant gives 
    helpful, detailed, and polite answers to the user's questions.
    Human: <image>
    Human: Translate the following audio description for these images from English to German even if the audio 
    description does not match the image: \n Back to Matthieu Fournier. The electric blue canopy of his paraglider contrasts with the white of the surrounding landscape.".
    AI: '''

    print(translate(image_files, query))
