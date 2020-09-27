import json

a = {'task_name': 'extract_features',
     'tasks':
         [
         {'image_loc': {'key': '08bfc10v7t.png', 'storage_type': 's3', 'bucket_name': 'spacer-test'},
          'feature_loc': {'key': 'dummy', 'storage_type': 'memory', 'bucket_name': null},
          'feature_extractor_name': 'vgg16_coralnet_ver1',
          'rowcols': [[20, 265], [76, 295], [59, 274], [151, 62]],
          'job_token': 'regression_job'}]
     }
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
msg = JobMsg.deserialize(a)
print(msg)
