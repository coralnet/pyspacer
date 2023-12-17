import unittest

from PIL import Image

from spacer.exceptions import RowColumnInvalidError, TrainingLabelsError
from spacer.messages import DataLocation, ImageLabels, TrainingTaskLabels
from spacer.task_utils import check_extract_inputs, preprocess_labels
from spacer.train_utils import make_random_data


class TestRowColChecks(unittest.TestCase):

    def test_ints(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(1.1, 1.2)]
        with self.assertRaises(RowColumnInvalidError):
            check_extract_inputs(img, rowcols, 'img')

    def test_ok(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(0, 0), (99, 99)]
        try:
            check_extract_inputs(img, rowcols, 'img')
        except AssertionError:
            self.fail("check_extract_inputs raised error unexpectedly")

    def test_negative(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(-1, -1)]
        with self.assertRaises(RowColumnInvalidError):
            check_extract_inputs(img, rowcols, 'img')

    def test_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(100, 100)]
        with self.assertRaises(RowColumnInvalidError):
            check_extract_inputs(img, rowcols, 'img')

    def test_row_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(100, 99)]

        with self.assertRaises(RowColumnInvalidError) as context:
            check_extract_inputs(img, rowcols, 'img')
        self.assertEqual(
            str(context.exception),
            "img: Row value 100 falls outside this image's"
            " valid range of 0-99.")

    def test_col_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(99, 100)]

        with self.assertRaises(RowColumnInvalidError) as context:
            check_extract_inputs(img, rowcols, 'img')
        self.assertEqual(
            str(context.exception),
            "img: Column value 100 falls outside this image's"
            " valid range of 0-99.")


class TestPreprocessLabels(unittest.TestCase):

    def test_train_ref_val_split(self):
        """
        Let pyspacer handle the train/ref/val split.
        """
        n_data = 20
        points_per_image = 10
        feature_dim = 5
        class_list = [1, 2]
        features_loc_template = DataLocation(storage_type='memory', key='')

        labels = preprocess_labels(make_random_data(
            n_data, class_list, points_per_image,
            feature_dim, features_loc_template,
        ))

        # 80% 10% 10%
        self.assertEqual(len(labels['train']), 16)
        self.assertEqual(len(labels['ref']), 2)
        self.assertEqual(len(labels['val']), 2)

    def test_train_ref_val_split_large(self):
        """
        This assumes the default batch size config value of 5000 labels.

        Ideally, later we'd be able to override config for specific tests,
        and then set an override for this test class so that it doesn't
        depend on the non-test config value.
        """
        n_data = 60
        points_per_image = 1000
        feature_dim = 5
        class_list = [1, 2]
        features_loc_template = DataLocation(storage_type='memory', key='')

        labels = preprocess_labels(make_random_data(
            n_data, class_list, points_per_image,
            feature_dim, features_loc_template,
        ))

        # 80%+ (capped to 5000 points) 10%
        self.assertEqual(len(labels['train']), 49)
        self.assertEqual(len(labels['ref']), 5)
        self.assertEqual(len(labels['val']), 6)

    def test_too_few_images(self):
        n_data = 2
        points_per_image = 5
        feature_dim = 5
        class_list = [1, 2]
        features_loc_template = DataLocation(storage_type='memory', key='')

        with self.assertRaises(TrainingLabelsError) as cm:
            preprocess_labels(make_random_data(
                n_data, class_list, points_per_image,
                feature_dim, features_loc_template,
            ))
        self.assertEqual(
            str(cm.exception),
            "The training data has 2 image(s), but need at least 3"
            " to populate train/ref/val sets.")

    def test_train_ref_1_common_class(self):
        points_per_image = 20
        feature_dim = 5
        features_loc_template = DataLocation(storage_type='memory', key='')

        with self.assertRaises(TrainingLabelsError) as cm:
            preprocess_labels(TrainingTaskLabels(
                train=make_random_data(
                    1, [1, 2], points_per_image,
                    feature_dim, features_loc_template,
                ),
                ref=make_random_data(
                    1, [2, 3], points_per_image,
                    feature_dim, features_loc_template,
                ),
                val=make_random_data(
                    1, [1, 2, 3], points_per_image,
                    feature_dim, features_loc_template,
                ),
            ))
        self.assertEqual(
            str(cm.exception),
            "Need multiple classes to do training. After preprocessing"
            " training data, there are 1 class(es) left.")

    def test_trainref_val_0_common_classes(self):
        points_per_image = 20
        feature_dim = 5
        features_loc_template = DataLocation(storage_type='memory', key='')

        with self.assertRaises(TrainingLabelsError) as cm:
            preprocess_labels(TrainingTaskLabels(
                # train+ref: 1, 2
                train=make_random_data(
                    1, [1, 2, 3], points_per_image,
                    feature_dim, features_loc_template,
                ),
                ref=make_random_data(
                    1, [1, 2], points_per_image,
                    feature_dim, features_loc_template,
                ),
                # val: 3
                val=make_random_data(
                    1, [3], points_per_image,
                    feature_dim, features_loc_template,
                ),
            ))
        self.assertEqual(
            str(cm.exception),
            "After preprocessing training data, 'val' set is empty.")

    def test_filter_by_accepted_classes(self):

        labels = preprocess_labels(
            TrainingTaskLabels(
                train=ImageLabels({
                    '1': [(100, 100, 1),
                          (200, 200, 2),
                          (300, 300, 3)],
                }),
                ref=ImageLabels({
                    '2': [(200, 200, 2),
                          (300, 300, 3),
                          (100, 100, 1)],
                }),
                val=ImageLabels({
                    '3': [(300, 300, 3),
                          (100, 100, 1),
                          (200, 200, 2)],
                }),
            ),
            accepted_classes={1, 2},
        )

        self.assertEqual(labels.train.data['1'], [(100, 100, 1), (200, 200, 2)])
        self.assertEqual(labels.ref.data['2'], [(200, 200, 2), (100, 100, 1)])
        self.assertEqual(labels.val.data['3'], [(100, 100, 1), (200, 200, 2)])


if __name__ == '__main__':
    unittest.main()
