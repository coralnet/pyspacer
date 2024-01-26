# Changelog

## 0.8.0 (WIP)

- `ImageFeatures` with `valid_rowcol=False` are no longer supported for training. For now they are still supported for classification.

- S3 downloads are now always performed in the main thread, to prevent `RuntimeError: cannot schedule new futures after interpreter shutdown`.

- `S3Storage` and `storage_factory()` now use the parameter name `bucket_name` instead of `bucketname` to be consistent with other usages in pyspacer (by @yeelauren).

- `URLStorage` downloads and existence checks now have an explicit timeout of 20 seconds (this is a timeout for continuous unresponsiveness, not for the whole response).

- EfficientNet feature extraction now uses CUDA if available (by @yeelauren).

- Updates to pip-install dependencies:

  - Pillow: >=10.0.1 to >=10.2.0

## 0.7.0

- `TrainClassifierMsg` labels arguments have changed. Instead of `train_labels` and `val_labels`, it now takes a single argument `labels`, which is a `TrainingTaskLabels` object (basically a set of 3 `ImageLabels` objects: training set, reference set, and validation set).

- The new function `task_utils.preprocess_labels()` can be called in advance of building a TrainClassifierMsg, to 1) split a single ImageLabels instance into reasonably-proportioned train/ref/val sets, 2) filter labels to only a desired set of classes, and 3) run error checks.

- Removed `MIN_TRAINIMAGES` config var. Minimum number of images for training is now 1 train set, 1 ref set, and 1 val set; or 3 total if leaving the split to pyspacer.

- Added `LOG_DESTINATION` and `LOG_LEVEL` config vars, providing configurable logging for test-suite runs or quick scripts.

- Logging statements throughout pyspacer's codebase now use module-name loggers rather than the root logger, allowing end-applications to keep their logs organized.

- Fixed bug where int config vars couldn't be configured through environment vars or secrets.json.

- Updated various error cases (mainly SpacerInputErrors, asserts, and ValueErrors) with more descriptive error classes. The `SpacerInputError` class is no longer available.

## 0.6.1

- In 0.5.0, the hash check when loading a feature extractor was broken in two ways. First, it got an error when trying to check the hash. Second, if the hash check failed for a remote-loaded extractor file, then a second attempt at loading would still allow extraction to proceed. This release fixes both problems.

## 0.6.0

- Fixed `DummyExtractor` constructor so that `data_locations` defaults to an empty dict, not an empty list. This fixes serialization of an `ExtractFeaturesMsg` containing `DummyExtractor`.

- Updates to pip-install dependencies:

  - Pillow: >=9.0.1 to >=10.0.1

## 0.5.0

- Generalized feature extractor support by allowing use of any `FeatureExtractor` subclass instance, and extractor files loaded from anywhere (not just from CoralNet's S3 bucket, which requires CoralNet auth).

- In `ExtractFeaturesMsg` and `ClassifyImageMsg`, the parameter `feature_extractor_name` (a string) has been replaced with `extractor` (a `FeatureExtractor` instance).

- In `ExtractFeaturesReturnMsg`, `model_was_cached` has been replaced by `extractor_loaded_remotely`, because now filesystem-caching doesn't apply to some extractor files (they may originally be from the filesystem).

- Config variable `LOCAL_MODEL_PATH` is now `EXTRACTORS_CACHE_DIR`. This is now used by any remote-loaded (S3 or URL based) extractor files. If extractor files are loaded from the filesystem, then it's now possible to run PySpacer without defining any config variable values.

- Added `AWS_REGION` config var, which is now required for S3 usage.

- Added `TEST_EXTRACTORS_BUCKET` and `TEST_BUCKET` config vars for unit tests, but these are not really usable by anyone besides core devs at the moment.

- Some raised errors' types have changed to PySpacer's own `ConfigError` or `HashMismatchError`, and there are cases where error-raising semantics/timing have changed slightly.

## 0.4.1

- Allowed configuration of `MAX_IMAGE_PIXELS`, `MAX_POINTS_PER_IMAGE`, and `MIN_TRAINIMAGES`.

- Previously, if `secrets.json` was present but missing a config value, then pyspacer would go on to look for that config value in Django settings. This is no longer the case; pyspacer now only respects at most one of secrets.json or Django settings (secrets take precedence).

- Updated repo URL from `beijbom/pyspacer` to `coralnet/pyspacer`.

## 0.4.0

- PySpacer now supports Python 3.8+ (testing against 3.8 - 3.10). Support for 3.6 and 3.7 has been dropped.

- Updates to pip-install dependencies:

  - Pillow: >=4.2.0 to >=9.0.1
  - numpy: >=1.17.5 to >=1.19
  - scikit-learn: ==0.22.1 to ==1.1.3 (loading models from older versions should still work)
  - torch: ==1.4.0 to ==1.13.1
  - torchvision: ==0.5.0 to ==0.14.1
  - Removed wget
  - Removed scikit-image
  - Removed tqdm (but it's still a developer requirement)
  - (Removed botocore from dependencies list, but only because it was redundant with boto3 already depending on it)

- Previously, config variables could be specified by a `secrets.json` file or by environment variables. Now a third way is available: a `SPACER` setting in a Django project. Also, the `secrets.json` method no longer uses `SPACER_` prefixes for each variable name. See README for details.

- The `LOCAL_MODELS_PATH` setting is now explicitly required. It was previously not required upfront, but its absence would make some tests fail.

- When an image is sourced from a URL, and the download fails, PySpacer now raises a `SpacerInputError` (instead of a `URLError` for example). The new `SpacerInputError` exception class indicates that the error was most likely caused by the input given to PySpacer (such as an unreachable URL) rather than by a PySpacer bug.

- PySpacer now only configures logging for fire / AWS Batch. When used as a pluggable app, it leaves the existing logging config alone.

## 0.3.1

Upgrade-relevant changes have not been carefully tracked up to this point. If you're unsure how to upgrade, consider starting your environment fresh from 0.4.0 or later.
