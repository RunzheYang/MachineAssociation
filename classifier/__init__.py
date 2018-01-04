from .lenet import LeNet

def get_classifier(model):
    if model == 'lenet':
        return LeNet()
    else:
        print("model %s doesn't exist." % (name))
        return None
