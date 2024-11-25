import random
import unittest
from unittest import mock

from PIL import Image

from spacer import config
from spacer.data_classes import (
    ImageFeatures, ImageLabels, LabelId, PointFeatures, ValResults)
from spacer.exceptions import (
    DataLimitError, RowColumnInvalidError, RowColumnMismatchError)
from spacer.extractors import DummyExtractor
from spacer.messages import (
    ClassifyFeaturesMsg,
    ClassifyImageMsg,
    ClassifyReturnMsg,
    DataLocation,
    ExtractFeaturesMsg,
    ExtractFeaturesReturnMsg,
    TrainClassifierMsg,
    TrainClassifierReturnMsg,
    TrainingTaskLabels,
)
from spacer.storage import (
    clear_memory_storage,
    load_classifier,
    S3Storage,
    storage_factory,
    store_classifier,
    store_image,
)
from spacer.tasks import (
    classify_features,
    classify_image,
    extract_features,
    train_classifier,
)
from spacer.task_utils import preprocess_labels
from spacer.tests.utils import cn_beta_fixture_location, temp_s3_filepaths
from spacer.train_utils import make_random_data, train
from .decorators import require_test_fixtures

TEST_URL = \
    'https://upload.wikimedia.org/wikipedia/commons/7/7b/Red_sea_coral_reef.jpg'


class TestImageAndPointLimitsAsserts(unittest.TestCase):

    def test_image_too_large(self):
        clear_memory_storage()
        img_loc = DataLocation(storage_type='memory', key='img')

        store_image(img_loc, Image.new('RGB', (10001, 10000)))
        msg = ExtractFeaturesMsg(
            job_token='test',
            extractor=DummyExtractor(),
            image_loc=img_loc,
            rowcols=[(1, 1)],
            feature_loc=DataLocation(storage_type='memory',
                                     key='feats')
        )
        with self.assertRaises(DataLimitError) as context:
            extract_features(msg)
        self.assertEqual(
            "Image img has 10001 x 10000 = 100010000 total pixels,"
            " which is larger than the max allowed of 100000000.",
            str(context.exception))

    def test_image_ok_size(self):
        clear_memory_storage()
        img_loc = DataLocation(storage_type='memory', key='img')

        store_image(img_loc, Image.new('RGB', (10000, 10000)))
        msg = ExtractFeaturesMsg(
            job_token='test',
            extractor=DummyExtractor(),
            image_loc=img_loc,
            rowcols=[(1, 1)],
            feature_loc=DataLocation(storage_type='memory',
                                     key='feats')
        )
        try:
            extract_features(msg)
        except AssertionError:
            self.fail("Image size assert tripped unexpectedly.")

    def test_too_many_points(self):
        clear_memory_storage()
        img_loc = DataLocation(storage_type='memory', key='img')

        store_image(img_loc, Image.new('RGB', (1000, 1000)))
        msg = ExtractFeaturesMsg(
            job_token='test',
            extractor=DummyExtractor(),
            image_loc=img_loc,
            rowcols=[(i, i) for i in range(config.MAX_POINTS_PER_IMAGE + 1)],
            feature_loc=DataLocation(storage_type='memory',
                                     key='feats')
        )
        with self.assertRaises(DataLimitError) as context:
            extract_features(msg)
        self.assertEqual(
            f"{config.MAX_POINTS_PER_IMAGE + 1} point locations were specified"
            f" for image img, and that's larger than"
            f" the max allowed of {config.MAX_POINTS_PER_IMAGE}.",
            str(context.exception))

    def test_ok_nbr_points(self):
        clear_memory_storage()
        img_loc = DataLocation(storage_type='memory', key='img')

        store_image(img_loc, Image.new('RGB', (1000, 1000)))
        msg = ExtractFeaturesMsg(
            job_token='test',
            extractor=DummyExtractor(),
            image_loc=img_loc,
            rowcols=[(i, i) for i in range(config.MAX_POINTS_PER_IMAGE)],
            feature_loc=DataLocation(storage_type='memory',
                                     key='feats')
        )
        try:
            extract_features(msg)
        except AssertionError:
            self.fail("Point count assert tripped unexpectedly.")


class TestExtractFeatures(unittest.TestCase):

    def test_default(self):

        clear_memory_storage()
        img_loc = DataLocation(storage_type='memory', key='img')

        store_image(img_loc, Image.new('RGB', (100, 100)))
        msg = ExtractFeaturesMsg(
            job_token='test',
            extractor=DummyExtractor(),
            image_loc=img_loc,
            rowcols=[(1, 1), (2, 2)],
            feature_loc=DataLocation(storage_type='memory',
                                     key='feats')
        )
        return_msg = extract_features(msg)
        self.assertEqual(type(return_msg), ExtractFeaturesReturnMsg)
        storage = storage_factory('memory')
        self.assertTrue(storage.exists('feats'))

    def test_duplicate_rowcols(self):

        msg = ExtractFeaturesMsg(
            job_token='job_nbr_1',
            extractor=DummyExtractor(),
            rowcols=[(100, 100), (50, 50), (100, 100)],
            image_loc=DataLocation(storage_type='memory',
                                   key='tmp_img'),
            feature_loc=DataLocation(storage_type='memory',
                                     key='tmp_feats')
        )
        store_image(msg.image_loc, Image.new('RGB', (110, 110)))
        _ = extract_features(msg)
        features = ImageFeatures.load(msg.feature_loc)
        self.assertEqual(len(features.point_features), len(msg.rowcols))


def spy_decorator(method_to_decorate):
    """
    A way to track calls to a class's instance method, across all instances
    of the class. From:
    https://stackoverflow.com/a/41599695
    """
    mock_obj = mock.MagicMock()
    def wrapper(self, *args, **kwargs):
        mock_obj(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)
    wrapper.mock_obj = mock_obj
    return wrapper


class TestTrainClassifier(unittest.TestCase):

    def do_basic_run(self, class_list, clf_type):

        # Parameters for data generation
        n_data = 200
        points_per_image = 20
        feature_dim = 5

        # Create training data.
        features_loc_template = DataLocation(storage_type='memory', key='')
        labels = preprocess_labels(
            make_random_data(
                n_data, class_list, points_per_image,
                feature_dim, features_loc_template,
            )
        )

        # Train once by calling directly so that we have a
        # previous classifier.
        clf, _ = train(
            labels.train, labels.ref, features_loc_template, 1, clf_type)

        previous_classifier_loc = DataLocation(storage_type='memory',
                                               key='pc')
        store_classifier(previous_classifier_loc, clf)

        valresult_loc = DataLocation(storage_type='memory', key='val_res')

        msg = TrainClassifierMsg(
            job_token='test',
            trainer_name='minibatch',
            nbr_epochs=1,
            clf_type=clf_type,
            labels=labels,
            features_loc=features_loc_template,
            previous_model_locs=[previous_classifier_loc],
            model_loc=DataLocation(storage_type='memory', key='model'),
            valresult_loc=valresult_loc
        )
        return_msg = train_classifier(msg)
        self.assertEqual(type(return_msg), TrainClassifierReturnMsg)

        # Do some checks on ValResults
        val_res = ValResults.load(valresult_loc)
        self.assertEqual(type(val_res), ValResults)
        self.assertEqual(len(val_res.gt), len(val_res.est))
        self.assertEqual(len(val_res.gt), len(val_res.scores))
        self.assertSetEqual(set(val_res.classes), set(class_list))

        self.assertEqual(
            len(val_res.gt), labels.val.label_count,
            msg="val_res has the correct number of labels")

    def test_lr_int_labels(self):
        self.do_basic_run([1, 2], 'LR')

    def test_mlp_int_labels(self):
        self.do_basic_run([1, 2], 'MLP')

    def test_lr_str_labels(self):
        self.do_basic_run(['Porites', 'CCA', 'Sand'], 'LR')

    def test_mlp_str_labels(self):
        self.do_basic_run(['Porites', 'CCA', 'Sand'], 'MLP')

    def test_row_column_matching(self):

        feature_dim = 5

        point_features = [
            PointFeatures(row=100, col=100, data=[1.5]*feature_dim),
            PointFeatures(row=50, col=50, data=[1.4]*feature_dim),
        ]
        features = ImageFeatures(
            point_features=point_features,
            valid_rowcol=True,
            feature_dim=feature_dim,
            npoints=len(point_features),
        )
        feature_loc = DataLocation(storage_type='memory', key='1.feats')
        features.store(feature_loc)
        train_labels = ImageLabels({
            '1.feats': [
                (100, 100, 1),
                (50, 50, 2),
                # Duplicate point; the row/column are still in features,
                # just not duplicated there
                (100, 100, 1),
            ]
        })
        ref_labels = make_random_data(
            1, [1, 2], 2, feature_dim,
            DataLocation(storage_type='memory', key='2.feats'))
        val_labels = make_random_data(
            1, [1, 2], 2, feature_dim,
            DataLocation(storage_type='memory', key='3.feats'))
        features_loc_template = DataLocation(storage_type='memory', key='')

        msg = TrainClassifierMsg(
            job_token='test',
            trainer_name='minibatch',
            nbr_epochs=1,
            clf_type='LR',
            labels=TrainingTaskLabels(
                train=train_labels,
                ref=ref_labels,
                val=val_labels,
            ),
            features_loc=features_loc_template,
            previous_model_locs=[],
            model_loc=DataLocation(storage_type='memory', key='model'),
            valresult_loc=DataLocation(storage_type='memory', key='result')
        )
        # Shouldn't get an error
        train_classifier(msg)

        msg.labels['train'] = ImageLabels({
            '1.feats': [
                (100, 100, 1),
                (50, 50, 2),
                # Row/column not in features
                (25, 25, 1),
            ]
        })

        with self.assertRaises(RowColumnMismatchError):
            train_classifier(msg)

    @staticmethod
    def do_feature_caching_test(feature_cache_dir):
        n_data = 3
        points_per_image = 10
        feature_dim = 5
        class_list = [1, 2]
        bucket_name = config.TEST_BUCKET

        # Remote storage.
        features_loc_template = DataLocation(
            storage_type='s3', key='', bucket_name=bucket_name)

        load_remote = spy_decorator(S3Storage._load_remote)

        with (
            mock.patch.object(
                S3Storage, '_load_remote', load_remote),
            temp_s3_filepaths(
                bucket_name=bucket_name,
                num_data_locations=n_data) as s3_feature_paths,
        ):
            msg = TrainClassifierMsg(
                job_token='test',
                trainer_name='minibatch',
                nbr_epochs=2,
                clf_type='MLP',
                labels=preprocess_labels(make_random_data(
                    n_data, class_list, points_per_image,
                    feature_dim, features_loc_template,
                    im_keys=s3_feature_paths,
                )),
                features_loc=features_loc_template,
                previous_model_locs=[],
                model_loc=DataLocation(storage_type='memory', key='model'),
                valresult_loc=DataLocation(storage_type='memory', key='valresult'),
                feature_cache_dir=feature_cache_dir,
            )
            train_classifier(msg)

        return load_remote.mock_obj

    @require_test_fixtures
    def test_feature_caching_enabled(self):
        load_remote_mock = self.do_feature_caching_test(
            TrainClassifierMsg.FeatureCache.AUTO)
        self.assertEqual(
            load_remote_mock.call_count, 3,
            "Should go like: download ref, download train, cache-load"
            " train, download val")

    @require_test_fixtures
    def test_feature_caching_disabled(self):
        load_remote_mock = self.do_feature_caching_test(
            TrainClassifierMsg.FeatureCache.DISABLED)
        self.assertEqual(
            load_remote_mock.call_count, 4,
            "Should go like: download ref, download train, download"
            " train, download val")


class ClassifyReturnMsgTest(unittest.TestCase):

    @staticmethod
    def _train_classifier(class_list):
        # Parameters for data generation
        n_data = 200
        points_per_image = 20
        feature_dim = 5

        # Create training data.
        features_loc_template = DataLocation(storage_type='memory', key='')
        labels = preprocess_labels(
            make_random_data(
                n_data, class_list, points_per_image,
                feature_dim, features_loc_template,
            )
        )

        model_loc = DataLocation(storage_type='memory', key='model')
        valresult_loc = DataLocation(storage_type='memory', key='val_res')

        msg = TrainClassifierMsg(
            job_token='test',
            trainer_name='minibatch',
            nbr_epochs=1,
            clf_type='MLP',
            labels=labels,
            features_loc=features_loc_template,
            previous_model_locs=[],
            model_loc=model_loc,
            valresult_loc=valresult_loc
        )
        train_classifier(msg)

        return model_loc

    def _validate_return_msg(self, return_msg, valid_rowcol):

        self.assertTrue(isinstance(return_msg.runtime, float))

        for row, col, scores in return_msg.scores:
            self.assertTrue(isinstance(scores, list))
            self.assertEqual(len(scores), len(return_msg.classes))

            if valid_rowcol:
                self.assertTrue(isinstance(return_msg[(row, col)], list))
                self.assertEqual(return_msg[(row, col)], scores)
                self.assertTrue(isinstance(row, int))
                self.assertTrue(isinstance(col, int))
            else:
                self.assertRaises(ValueError, return_msg.__getitem__, (10, 20))
                self.assertIsNone(row)
                self.assertIsNone(col)

        for class_ in return_msg.classes:
            self.assertTrue(isinstance(class_, LabelId))

        self.assertTrue(isinstance(return_msg.valid_rowcol, bool))

        self.assertEqual(type(return_msg), ClassifyReturnMsg)


class TestClassifyFeatures(ClassifyReturnMsgTest):

    def setUp(self):
        config.filter_warnings()

    @require_test_fixtures
    def test_legacy(self):
        msg = ClassifyFeaturesMsg(
            job_token='my_job',
            feature_loc=cn_beta_fixture_location('example.jpg.feats'),
            classifier_loc=cn_beta_fixture_location('example.model')
        )

        return_msg = classify_features(msg)
        self._validate_return_msg(return_msg, False)

    @require_test_fixtures
    def test_new(self):

        feats = ImageFeatures.make_random([1, 2, 3, 2], feature_dim=4096)
        feature_loc = DataLocation(storage_type='memory',
                                   key='new.jpg.feats')
        feats.store(feature_loc)

        model_loc = cn_beta_fixture_location('example.model')

        msg = ClassifyFeaturesMsg(
            job_token='my_job',
            feature_loc=feature_loc,
            classifier_loc=model_loc
        )

        return_msg = classify_features(msg)

        self._validate_return_msg(return_msg, True)

    def do_train_and_classify(self, class_list):
        model_loc = self._train_classifier(class_list)

        feats = ImageFeatures.make_random(
            random.choices(class_list, k=4), feature_dim=5)
        feature_loc = DataLocation(storage_type='memory',
                                   key='new.jpg.feats')
        feats.store(feature_loc)

        msg = ClassifyFeaturesMsg(
            job_token='my_job',
            feature_loc=feature_loc,
            classifier_loc=model_loc
        )

        return_msg = classify_features(msg)

        self._validate_return_msg(return_msg, True)

    def test_train_and_classify_int_labels(self):
        self.do_train_and_classify([1, 2, 3])

    def test_train_and_classify_str_labels(self):
        self.do_train_and_classify(['Porites', 'CCA', 'Sand'])


class TestClassifyImage(ClassifyReturnMsgTest):

    def setUp(self):
        config.filter_warnings()

    @require_test_fixtures
    def test_deploy_simple(self):
        msg = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            extractor=DummyExtractor(),
            rowcols=[(100, 100), (200, 200)],
            classifier_loc=cn_beta_fixture_location('example.model')
        )
        return_msg = classify_image(msg)
        self._validate_return_msg(return_msg, True)

    def do_train_and_classify(self, class_list):
        model_loc = self._train_classifier(class_list)

        msg = ClassifyImageMsg(
            job_token='my_job',
            extractor=DummyExtractor(feature_dim=5),
            rowcols=[(100, 100), (200, 200)],
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            classifier_loc=model_loc
        )

        return_msg = classify_image(msg)

        self._validate_return_msg(return_msg, True)

    def test_train_and_classify_int_labels(self):
        self.do_train_and_classify([1, 2, 3])

    def test_train_and_classify_str_labels(self):
        self.do_train_and_classify(['Porites', 'CCA', 'Sand'])


class TestClassifyImageCache(unittest.TestCase):

    def setUp(self):
        config.filter_warnings()

    @require_test_fixtures
    def test_classify_image_with_caching(self):
        """ Call classify_image three times.
        The first 2 time with same message.
        The last time with a new message (different classifier).
        Due to caching, the second call should be the fastest of the three.
        """

        load_classifier.cache_clear()
        msg = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            extractor=DummyExtractor(),
            rowcols=[(100, 100), (200, 200)],
            classifier_loc=cn_beta_fixture_location('example.model')
        )

        msg2 = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            extractor=DummyExtractor(),
            rowcols=[(100, 100), (200, 200)],
            classifier_loc=cn_beta_fixture_location('example_model2.pkl')
        )

        return_msg1 = classify_image(msg)
        return_msg2 = classify_image(msg)
        return_msg3 = classify_image(msg2)
        self.assertLess(return_msg2.runtime, return_msg1.runtime)
        self.assertLess(return_msg2.runtime, return_msg3.runtime)


class TestClassifyBadRowcols(unittest.TestCase):

    @require_test_fixtures
    def test_image_classify(self):
        msg = ClassifyImageMsg(
            job_token='my_job',
            image_loc=DataLocation(storage_type='url',
                                   key=TEST_URL),
            extractor=DummyExtractor(),
            rowcols=[(-1, -1)],
            classifier_loc=cn_beta_fixture_location('example.model')
        )
        with self.assertRaises(RowColumnInvalidError) as context:
            classify_image(msg)
        self.assertEqual(
            "https://upload.wikimedia.org/wikipedia/commons/7/7b/"
            "Red_sea_coral_reef.jpg: Row values must be non-negative."
            " Given value was: -1",
            str(context.exception))


if __name__ == '__main__':
    unittest.main()
