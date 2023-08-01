import torch
from vap.modules.lightning_module import VAPModule

CHECKPOINT = "example/checkpoints/checkpoint.ckpt"

if __name__ == "__main__":

    #module = VAPModule.load_from_checkpoint(CHECKPOINT)
    model = VAPModule.load_model(CHECKPOINT)

    device = 'cpu'
    if torch.cuda.is_available():
        model = model.to('cuda')
        device = 'cuda'
        print('Using CUDA')

    x = torch.randn((1,2,32000))
    out = model.probs(x.to(device))
    print('out: ', out.keys())
    print('p_now: ', out['p_now'].shape)
    print('p_fut: ', out['p_future'].shape)
