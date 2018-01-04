from .lenet import LeNet

def get_classifier(name):
    if name == 'lenet':
        return LeNet()
    else:
        print("model %s doesn't exist." % (name))
        return None
