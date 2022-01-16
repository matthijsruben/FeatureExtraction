import logging

from src.features.contour import get_contour_features
from src.features.medianwidth import get_median_width_features
from src.features.slantness import get_slantness_features
from src.features.writingzones import get_writing_zone_features
from src.iamloader import load_data_as_tuples
from src.models import TextImage
from src.writer import write_features


def extract_and_write_features():
    data = load_data_as_tuples('data/lines/lines.tgz')
    images = []

    for datapoint in data[:5]:
        text_image = TextImage(datapoint[0], datapoint[1])
        text_image.add_features(get_writing_zone_features(text_image))
        text_image.add_features(get_median_width_features(text_image))
        lc_features, uc_features = get_contour_features(text_image)
        text_image.add_features(lc_features)
        text_image.add_features(uc_features)
        text_image.add_features(get_slantness_features(text_image))
        images.append(text_image)

    write_features(images, "./features.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_and_write_features()
