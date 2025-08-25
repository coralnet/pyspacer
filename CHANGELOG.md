# Changelog

## 0.13.0 (WIP)

- Updates to pip-install dependencies:

  - torch: >=2.6,<2.7 to >=2.6,<2.9
  - torchvision: >=0.21,<0.22 to >=0.21,<0.24

- Fixed an `AttributeError` in `TorchExtractor.load_weights()` that would happen on numpy 1.26.1 through 1.26.5. These are the versions where `numpy._core` exists but doesn't have attributes available; on these versions we now access `numpy.core` like on other 1.x versions.

- AWS related changes:

  - Added `AWS_PROFILE_NAME` and `AWS_SESSION_TOKEN` config variables to support more ways of accessing AWS.
  - Moved `get_s3_resource()` from `config.py` to a new file, `aws.py`.
  - Added an `aws_check()` function in `aws.py`, to help debug AWS configuration.
  - `S3Storage.exists()` now only returns False if there is a ClientError with status code 404. In other ClientError cases, including 403, this method will now re-raise the error. Due to this, it's now recommended to ensure you have the ListBucket permission for any S3 bucket you work with; otherwise, S3 will return 403 instead of 404 for a missing file (see [this article](https://repost.aws/articles/ARe3OTZ3SCTWWqGtiJ6aHn8Q/why-does-s-3-return-403-instead-of-404-when-the-object-doesnt-exist)).

## 0.12.0

- Updates to pip-install dependencies:

  - torch: >=2.2,<2.5 to >=2.6,<2.7
  - torchvision: >=0.17,<0.20 to >=0.21,<0.22

- `URLStorage` downloads: if a `TimeoutError` occurs while calling `read()` on the response, the error will be wrapped in a `spacer.exceptions.URLDownloadError`. Previously this was only the case for the `urlopen()` call, not the `read()` call.

## 0.11.0

- Feature extractor class changes:

  - `FeatureExtractor` and its built-in subclasses should now be imported like `from spacer.extractors import <class>` instead of `from spacer.extract_features import <class>`.

  - High-level usage of `FeatureExtractor` instances is the same as before - invoking `__call__()` performs feature extraction on an image. However, subclass implementations should now generally define a `patches_to_features()` method instead of overriding `__call__()`.

  - There is now a `TorchExtractor` class which has details that are specific to PyTorch but not to EfficientNet. So, it's suitable as a starting point for a custom PyTorch extractor that uses another type of network. `EfficientNetExtractor` now inherits from TorchExtractor.

  - There are now `CROP_SIZE` and `BATCH_SIZE` class-level variables available.

- Python 3.12 support added, so the supported versions are now 3.10-3.12.

- Updates to pip-install dependencies:

  - boto3: >=1.26.0 to >=1.26.115
  - Pillow: >=10.2.0 to >=10.4.0
  - numpy: >=1.21.4,<2 to >=1.22,<2.2
  - scikit-learn: ==1.1.3 to ==1.5.2 (loading models from previously-supported versions should still work)
  - torch: >=1.13.1,<2.3 to >=2.2,<2.5
  - torchvision: >=0.14.1,<0.18 to >=0.17,<0.20
  - tqdm: no longer required in any environment

  If you use numpy>=2.0, you probably also need torch>=2.3, torchvision>=0.18, and scipy>=1.13 (scipy is required by scikit-learn). Otherwise, you may get errors like "_ARRAY_API not found" and warnings like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.3 as it may crash."

  Also note that torch>=2.3 does not provide binaries for macOS x86.

- Feature extraction should now be able to tolerate more image color modes. Previously, `LA` and possibly other modes supported by Pillow would make feature extraction crash. (Note that all modes are converted to RGB for feature extraction purposes.)

- Config and test changes:

  - Some former usages of `TEST_BUCKET` have been changed to `CN_FIXTURES_BUCKET`, to more clearly denote test fixtures that are currently only available to CoralNet devs.

  - The remaining usages of `TEST_BUCKET` are now usable by anyone with an AWS account. This can be any S3 bucket that you have read and write access to.

  - `TEST_EXTRACTORS_BUCKET` is now known as `CN_TEST_EXTRACTORS_BUCKET`, again denoting fixtures currently only available to CoralNet devs.

  Related to these changes, now more tests are runnable without needing CoralNet AWS credentials. More tests are runnable in GitHub Actions CI, as well (even though that doesn't use AWS at all).

- Most of the repo's standalone scripts have been removed, thus avoiding confusion about their purpose.

## 0.10.0

- AWS credentials can now be obtained through the following methods, in addition to spacer config values as before:
  
  - AWS's metadata service (STS); proof of concept from @michaelconnor00
  - boto's auto-detection logic when neither of STS or spacer config are used (this was intended to work before, but needed fixing)

- Updates to pip-install dependencies:

  - numpy: >=1.19 to >=1.21.4,<2
  - boto3: nothing to >=1.26.0

## 0.9.0

- Python 3.8 and 3.9 support have been dropped; Python 3.11 support has been added.

- torch and torchvision accepted versions have been relaxed to accommodate Python 3.11. (torch==1.13.1 to torch>=1.13.1,<2.3; torchvision==0.14.1 to torchvision>=0.14.1,<0.18)

- `task_utils.preprocess_labels()` now has three available modes on how to split training annotations between train, ref, and val sets. Differences between the three modes - `VECTORS`, `POINTS`, and `POINTS_STRATIFIED` - are explained in the `SplitMode` Enum's comments. Additionally, all three modes now ensure that the ordering of the given training data has no effect on which data goes into train, ref, and val.

  The table below compares the three modes to the splitting functionality of earlier versions of pyspacer. Note that it's still possible to split train/ref/val yourself instead of letting pyspacer do it.

  | Mode              | Sets split in pyspacer | Order agnostic | Vectors can be split | Stratifies by label |
  |-------------------|------------------------|----------------|----------------------|---------------------|
  | 0.6.1 and earlier | Train/ref              | No             | No                   | No                  |
  | 0.7.0 - 0.8.0     | Train/ref/val          | No             | No                   | No                  |
  | VECTORS           | Train/ref/val          | Yes            | No                   | No                  |
  | POINTS            | Train/ref/val          | Yes            | Yes                  | No                  |
  | POINTS_STRATIFIED | Train/ref/val          | Yes            | Yes                  | Yes                 |

- The `train_classifier` task now accepts label IDs as either integers or strings, not just integers.

- The `train_classifier` task is now able to locally cache feature vectors which were loaded from remote storage, which can greatly speed up training from epoch 2 onward. This is optional and enabled by default; the location of the cache directory is also configurable.

## 0.8.0

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
