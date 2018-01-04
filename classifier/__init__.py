from .mnist_cnn import MnistCNN

def get_classifier(name):
    if name == 'mnist_cnn':
        return MnistCNN()
    else:
        print("model %s doesn't exist." % (name))
        return None
