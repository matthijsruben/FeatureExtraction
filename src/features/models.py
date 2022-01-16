from abc import ABC, abstractmethod


class Features(ABC):
    @property
    @abstractmethod
    def features(self):
        pass

    @property
    def feature_names(self):
        return self.features.keys()

    @property
    def feature_values(self):
        return self.features.values()


class ContourFeatures(Features):
    def __init__(self,
                 slant,
                 slant_mse,
                 local_max_freq,
                 local_min_freq,
                 max_slopes_left_avg,
                 max_slopes_right_avg,
                 min_slopes_left_avg,
                 min_slopes_right_avg,
                 lower):
        """
        Creates a ContourFeatures object.
        :param slant: slant of the regression line fit to the contour
        :param slant_mse: mean squared error between the regression line and the original characteristic contour
        :param local_max_freq: frequency of maxima in the lower contour
        :param local_min_freq: frequency of minima in the lower contour
        :param max_slopes_left_avg: average local slope to the left of local maxima in the contour
        :param max_slopes_right_avg: average local slope to the right of local maxima in the contour
        :param min_slopes_left_avg: average local slope to the left of local minima in the contour
        :param min_slopes_right_avg: average local slope of characteristic contour to the right of local minima in
        the contour
        :param lower: whether these features concern features for the lower characteristic contour (True) or the
        upper characteristic contour (False)
        """
        self.slant = slant
        self.slant_mse = slant_mse
        self.local_max_freq = local_max_freq
        self.local_min_freq = local_min_freq

        self.max_slopes_left_avg = max_slopes_left_avg
        self.max_slopes_right_avg = max_slopes_right_avg

        self.min_slopes_left_avg = min_slopes_left_avg
        self.min_slopes_right_avg = min_slopes_right_avg

        self.lower = lower

        feature_names = ["slant", "slant_mse", "local_max_freq", "local_min_freq", "max_slopes_left_avg",
                         "max_slopes_right_avg", "min_slopes_left_avg", "min_slopes_right_avg"]
        # It is not possible to use a lambda function here due to multithreading issues
        self._feature_names = map(self.prefix_feature_names, feature_names)

    @property
    def features(self):
        return {name: value for name, value in zip(self._feature_names, self.feature_values)}

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def feature_values(self):
        return [self.slant, self.slant_mse, self.local_max_freq, self.local_min_freq, self.max_slopes_left_avg,
                self.max_slopes_right_avg, self.min_slopes_left_avg, self.min_slopes_right_avg]

    def prefix_feature_names(self, name):
        prefix = "lc" if self.lower else "uc"
        return "{}_{}".format(prefix, name)


class MedianWidthFeatures(Features):
    def __init__(self, median_width):
        self.median_width = median_width

    @property
    def features(self):
        return {'median_width': self.median_width}


class WritingZoneFeatures(Features):
    def __init__(self, upper_zone, middle_zone, lower_zone):
        self.upper_zone = upper_zone
        self.middle_zone = middle_zone
        self.lower_zone = lower_zone

    @property
    def features(self):
        return {
            'upper_zone': self.upper_zone,
            'middle_zone': self.middle_zone,
            'lower_zone': self.lower_zone
        }


class SlantnessFeatures(Features):
    def __init__(self, max_angle, avg_angle, stdev_angle):
        self.max_angle = max_angle
        self.avg_angle = avg_angle
        self.stdev_angle = stdev_angle

    @property
    def features(self):
        return {
            'slantness_max_angle': self.max_angle,
            'slantness_avg_angle': self.avg_angle,
            'slantness_stdev_angle': self.stdev_angle
        }
