
class Feature(object):
    FEATURE_KEYS = ['input']
    LABEL_KEY = 'labels'
    NUM_CLASSES = 2

    @staticmethod
    def transformed_name(key):
        return key + '_xf'
