"""One FLUX.2-dev background inference per source-condition pair."""
from PIL import Image


class FluxBackground:
    def __init__(self, model_path, config):
        import torch
        from diffusers import Flux2Pipeline
        dtype = torch.bfloat16 if config['dtype'] == 'bf16' else torch.float16
        torch.cuda.empty_cache()
        self.pipe = Flux2Pipeline.from_pretrained(str(model_path), torch_dtype=dtype, local_files_only=True)
        if config['cpu_offload']:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to('cuda')

    def generate(self, image, prompt, config):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device).manual_seed(config['seed'])
        with torch.inference_mode():
            background = self.pipe(image=image, prompt=prompt,
                                   num_inference_steps=config['steps'],
                                   guidance_scale=config['guidance_scale'],
                                   generator=generator).images[0]
        if background.size != image.size:
            background = background.resize(image.size, Image.Resampling.LANCZOS)
        return background.convert('RGB')
