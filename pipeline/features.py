
class Feature(object):
    """
    Defines constants related to dataset features.
    This includes raw feature keys, the label key, number of classes,
    and a utility method for naming transformed features.
    Using a class like this helps maintain consistency across different
    parts of the pipeline (preprocessing, training, evaluation).
    """

    # List of raw numeric feature keys from the input data.
    NUMERIC_FEATURE_KEYS = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]

    # List of raw categorical feature keys from the input data.
    CATEGORICAL_FEATURE_KEYS = ['Wilderness_Area', 'Soil_Type']

    # Raw key for the label (target variable).
    LABEL_KEY = 'Cover_Type'

    # Number of unique classes for the classification task.
    NUM_CLASSES = 7

    @staticmethod
    def transformed_name(key: str) -> str:
        """
        Generates the name for a feature after it has been transformed by TFT.
        A common convention is to append '_xf'.

        Args:
            key: The original (raw) feature key.

        Returns:
            The transformed feature key.
        """
        return key + '_xf'
