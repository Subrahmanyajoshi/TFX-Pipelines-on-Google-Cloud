
class Feature(object):
    FEATURE_KEYS = ['input']
    LABEL_KEY = 'labels'

    @staticmethod
    def transformed_name(key):
        return key + '_xf'
