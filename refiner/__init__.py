from .unet import UNet


def get_classifier(model):
    if model == 'unet':
        return UNet()
    else:
        print("model %s doesn't exist." % (model))
        return None
