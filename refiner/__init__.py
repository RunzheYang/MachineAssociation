from .unet import UNet


def get_refiner(model):
    if model == 'unet':
        return UNet()
    else:
        print("model %s doesn't exist." % (model))
        return None
