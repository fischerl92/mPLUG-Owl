import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
# from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor
from mplug_owl.processing_mplug_owl import MplugOwlProcessor
import torch

import logging
logging.basicConfig(level=logging.INFO)

pretrained_ckpt_video = 'MAGAer13/mplug-owl-llama-7b-video'
pretrained_ckpt_multilingual = 'MAGAer13/mplug-owl-bloomz-7b-multilingual'

model = AutoModel.from_pretrained(
    pretrained_ckpt_multilingual,
    torch_dtype=torch.bfloat16,
    # device_map={'': 0},
)
logging.info("loaded model")

image_processor = AutoImageProcessor.from_pretrained(pretrained_ckpt_video)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt_multilingual)
processor = MplugOwlProcessor(image_processor, tokenizer)

logging.info("loaded everything else")

# We use a human/AI template to organize the context as a multi-turn conversation.
# <|video|> denotes an video placehold.
prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: What is the man doing in the video?
AI: ''']

video_list = ['/net/cephfs/shares/iict-sp4.ebling.cl.uzh/data/icare_2023_04/dataset4/access-services-swisstxt_ad-screenrec-batch-4-5_2023-03-16_1130/AD_Moskau_einfach/403_1-09-47.280000_1-09-49.400000.mp4']

# generate kwargs (the same in transformers) can be passed in the do_generate()
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}
inputs = processor(text=prompts, videos=video_list, num_frames=4, return_tensors='pt')
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    res = model.generate(**inputs, **generate_kwargs)
sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
print(sentence)
