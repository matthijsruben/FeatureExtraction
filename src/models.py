from src import utils
from src.features.models import Features


class TextImage:
    def __init__(self, writer, image):
        self.writer = writer
        self.image = image
        self.features = []

        self._bw_image = None

    @property
    def bw_image(self):
        """
        Returns a black-and-white (binary) version of the image; creates such version when this is requested for the
        first time.
        :return: A black-and-white version of the image.
        """
        if not self._bw_image:
            self._bw_image = utils.get_bw_image(self.image, 150)

        return self._bw_image

    def get_features_dictionary(self):
        """
        :return: A dictionary of feature names mapped to their respective values.
        """
        feature_dictionaries = [feature_type.features for feature_type in self.features]
        return {feature_name: feature_value for d in feature_dictionaries for feature_name, feature_value in d.items()}

    def add_features(self, features: Features):
        self.features.append(features)

    def as_dict(self):
        """
        :return: A dictionary of the writer of this image and all feature names mapped to their respective values.
        """
        return {**{"writer": self.writer}, **self.get_features_dictionary()}
