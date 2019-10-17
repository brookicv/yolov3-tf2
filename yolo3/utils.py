from functools import reduce
import numpy as np
from PIL import Image

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a,**kw)),funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")

def letterbox_image(image,size):
    """
    resize image with unchanged aspect ratio using padding
    """
    iw,ih = image.size
    w,h = size
    scale = min(w / iw,h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw,nh),Image.BICUBIC)
    new_image = Image.new("RGB",(128,128,128))
    new_image.paste(image,((w - nw) // 2,(h - nh) // 2))
    return new_image