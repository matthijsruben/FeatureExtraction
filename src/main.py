import concurrent.futures
import logging
import os
import sys
import time

from src.features.contour import get_contour_features
from src.features.medianwidth import get_median_width_features
from src.features.slantness import get_slantness_features
from src.features.writingzones import get_writing_zone_features
from src.iamloader import load_data_as_tuples
from src.models import TextImage
from src.writer import write_or_append_features

DEFAULT_OUTPUT_DIR = "./output/features"


def extract_features(datapoint):
    writer = datapoint[0]
    filename = datapoint[1][0]
    image = datapoint[1][1]

    logging.debug("Processing image {}".format(filename))

    text_image = TextImage(writer, filename, image)
    text_image.add_features(get_writing_zone_features(text_image))
    text_image.add_features(get_median_width_features(text_image))
    lc_features, uc_features = get_contour_features(text_image)
    text_image.add_features(lc_features)
    text_image.add_features(uc_features)
    text_image.add_features(get_slantness_features(text_image))

    return text_image


def extract_and_write_features(output_dir_abs):
    data = load_data_as_tuples('data/lines/lines.tgz', 'data/lines/xml')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = "{}/{}_features.csv".format(output_dir_abs, timestamp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        images = []

        for datapoint in data[:10]:
            futures.append(executor.submit(extract_features, datapoint))
        for future in concurrent.futures.as_completed(futures):
            images.append(future.result())

            if (len(images) % 10) == 0:
                # Periodically write features to prevent data loss when processing fails late in the process
                write_or_append_features(images, output_path)
                images.clear()

        write_or_append_features(images, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 1:
        logging.info("Script ran without arguments")
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        logging.info("Script ran with arguments: {}".format(" ".join(sys.argv)))
        output_dir = sys.argv[1].strip()

    if output_dir.endswith("/"):
        # Remove leading slash
        output_dir = output_dir[-1]

    logging.info("Configured to output to directory: {}".format(output_dir))
    output_dir = os.path.abspath(output_dir)
    logging.info("Absolute path of output directory: {}".format(output_dir))

    if not os.path.isdir(output_dir):
        logging.info("Directory {} does not yet exist. Creating...".format(output_dir))
        os.makedirs(output_dir, exist_ok=True)

    extract_and_write_features(output_dir)
