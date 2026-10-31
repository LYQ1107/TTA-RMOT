"""New choice/SOM paths; the existing YesTrack binary backend is unmodified."""
import hashlib
import re
import time
from PIL import ImageDraw,ImageFont

LETTERS='ABCDEFGHIJKLMNOPQR'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def choice_prompt(queries,coord):
    if len(queries)!=17:raise ValueError('fixed17 options required')
    choices='\n'.join(f'{LETTERS[j]}. {q["query_text"]}' for j,q in enumerate(queries))
    return (f'The crop depicts one tracked person at normalized position <{coord}>. '
        'Which one of the following descriptions matches this person?\n'+choices+
        '\nR. none of these\nAnswer with exactly one letter from A to R.')

def som_prompt(sentence):
    return f"Which numbered people match: {sentence}? Answer with the numbers separated by commas, or 'none'."

def parse_som(text,count):
    value=text.strip().rstrip('.').strip()
    if value.lower()=='none':return dict(valid=True,numbers=[],duplicate_numbers=0,error=None)
    if not re.fullmatch(r'[1-9]\d*(?:\s*,\s*[1-9]\d*)*',value):
        return dict(valid=False,numbers=None,duplicate_numbers=0,error='INVALID_FORMAT')
    numbers=list(map(int,re.split(r'\s*,\s*',value)))
    if any(n>count for n in numbers):return dict(valid=False,numbers=None,duplicate_numbers=0,error='OUT_OF_RANGE')
    return dict(valid=True,numbers=sorted(set(numbers)),duplicate_numbers=len(numbers)-len(set(numbers)),error=None)

def draw_som(image,candidates):
    result=image.copy();draw=ImageDraw.Draw(result);font=ImageFont.truetype(FONT,24)
    for j,c in enumerate(candidates,1):
        x,y,w,h=c['bbox_xywh'];draw.rectangle((x,y,x+w,y+h),outline='red',width=2)
        text=str(j);box=draw.textbbox((0,0),text,font=font);tw,th=box[2]-box[0]+6,box[3]-box[1]+6
        lx=max(0,min(result.width-tw,int(x)));ly=max(0,min(result.height-th,int(y)))
        draw.rectangle((lx,ly,lx+tw,ly+th),fill='white')
        draw.text((lx+3-box[0],ly+3-box[1]),text,font=font,fill='black')
    return result

def chat_inputs(backend,images,prompts):
    messages=[[dict(role='user',content=[dict(type='image',image=im),dict(type='text',text=prompt)])]
        for im,prompt in zip(images,prompts)]
    backend.processor.tokenizer.padding_side='left'
    return backend.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',padding=True).to(backend.device)

def choice_forward(backend,images,prompts):
    import torch
    ids=[]
    for letter in LETTERS:
        tokens=backend.processor.tokenizer.encode(letter,add_special_tokens=False)
        if len(tokens)!=1:raise ValueError('choice letter must have exactly one token: '+letter)
        ids.append(tokens[0])
    backend.capture=[];torch.cuda.synchronize(backend.device);start=time.perf_counter()
    with torch.inference_mode():
        inputs=chat_inputs(backend,images,prompts);out=backend.model(**inputs)
        logits=out.logits[:,-1,ids].float();p=logits.softmax(-1).cpu().tolist();raw=logits.cpu().tolist()
        if not torch.isfinite(logits).all():raise FloatingPointError('non-finite choice logits')
    torch.cuda.synchronize(backend.device);elapsed=time.perf_counter()-start;backend.check_versions()
    return [dict(option_logits=r,option_probabilities=x,option_token_ids=ids,
        argmax_letter=LETTERS[max(range(18),key=lambda j:x[j])]) for r,x in zip(raw,p)],elapsed

def som_forward(backend,image,prompt,count):
    import torch
    backend.capture=[];torch.cuda.synchronize(backend.device);start=time.perf_counter()
    with torch.inference_mode():
        inputs=chat_inputs(backend,[image],[prompt]);length=inputs['input_ids'].shape[1]
        out=backend.model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
        tokens=out[0,length:].cpu().tolist();text=backend.processor.tokenizer.decode(tokens,skip_special_tokens=True)
    torch.cuda.synchronize(backend.device);elapsed=time.perf_counter()-start;backend.check_versions()
    return dict(raw_text=text,generated_token_ids=tokens,generated_tokens=len(tokens),
        token_cap_reached=len(tokens)>=64,**parse_som(text,count)),elapsed
