
class Feature(object):
    NUMERIC_FEATURE_KEYS = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    CATEGORICAL_FEATURE_KEYS = ['Wilderness_Area', 'Soil_Type']
    LABEL_KEY = 'Cover_Type'
    NUM_CLASSES = 7

    @staticmethod
    def transformed_name(key):
        return key + '_xf'
