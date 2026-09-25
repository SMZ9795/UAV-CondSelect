"""Independent 512x288 Klein Base comparison: preparation, inference and blending."""
from pathlib import Path
import argparse,json
from PIL import Image,ImageFilter

def prepare(source,protection,output):
 output.mkdir(parents=True,exist_ok=True)
 original=Image.open(source).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
 hard=Image.open(protection).convert('L').resize((1024,576),Image.Resampling.NEAREST).point(lambda v:255 if v>127 else 0)
 soft=hard.filter(ImageFilter.GaussianBlur(28));masked=Image.composite(original.filter(ImageFilter.GaussianBlur(60)),original,soft)
 original.save(output/'source.jpg',quality=95);soft.save(output/'soft_mask.png');masked.save(output/'input.jpg',quality=95)

def blend(source,mask,background,output):
 source=Image.open(source).convert('RGB').resize((512,288),Image.Resampling.LANCZOS)
 mask=Image.open(mask).convert('L').resize((512,288),Image.Resampling.NEAREST)
 background=Image.open(background).convert('RGB').resize((512,288),Image.Resampling.LANCZOS)
 output.parent.mkdir(parents=True,exist_ok=True);Image.composite(source,background,mask).save(output)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
 q=sub.add_parser('prepare');q.add_argument('--source',type=Path,required=True);q.add_argument('--protection',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('blend');q.add_argument('--source',type=Path,required=True);q.add_argument('--mask',type=Path,required=True);q.add_argument('--background',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('generate');q.add_argument('--records',type=Path,default=Path(__file__).with_name('records.json'));q.add_argument('--assets',type=Path,required=True);q.add_argument('--model',required=True);q.add_argument('--revision',required=True);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.stage=='prepare':prepare(a.source,a.protection,a.output)
 elif a.stage=='blend':blend(a.source,a.mask,a.background,a.output)
 else:
  import torch
  from diffusers import Flux2KleinPipeline
  pipe=Flux2KleinPipeline.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16);pipe.enable_model_cpu_offload();a.output.mkdir(parents=True,exist_ok=True)
  for r in json.loads(a.records.read_text()):
   image=Image.open(a.assets/r['input_conditioning']).convert('RGB').resize((512,288),Image.Resampling.LANCZOS)
   with torch.inference_mode():result=pipe(image=image,prompt=r['prompt'],width=512,height=288,num_inference_steps=28,guidance_scale=3.0,generator=torch.Generator(device='cpu').manual_seed(43),text_encoder_out_layers=(9,18,27),output_type='pil').images[0]
   dest=a.output/(r['source_id']+'_'+r['condition']);dest.mkdir(parents=True,exist_ok=True)
   result.convert('RGB').save(dest/'background.png')
   blend(a.assets/r['source'],a.assets/r['mask'],dest/'background.png',dest/'final.png')
if __name__=='__main__':main()
