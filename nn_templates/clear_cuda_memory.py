del model

import gc, torch
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()
