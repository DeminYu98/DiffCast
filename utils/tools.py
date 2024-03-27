import logging


def exists(x):
    return x is not None       
        
def cycle(dl):
    while True:
        for data in dl:
            yield data

def print_log(message, is_main_process=True):
    if is_main_process:
        print(message)
        logging.info(message)

def show_img_info(imgs):
    print("="*30 + "Image Info" + "="*30)
    print("Tensor Shape: ",imgs.shape)
    print("Tensor DType: ",imgs.dtype)
    print("Max Value:", imgs.max())
    print("Min Value: ",imgs.min())
    print("Mean: ", imgs.mean())
        
