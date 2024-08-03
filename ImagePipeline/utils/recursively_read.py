import os

def recursively_read(rootdir, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    print(rootdir)
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts):
                out.append(os.path.join(r, file))
    return out