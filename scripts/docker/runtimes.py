from spacer.messages import JobReturnMsg, DataLocation, JobMsg
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer.storage import store_image, load_image
from spacer import config
from datetime import datetime
from spacer.tasks import process_job


def make_job(nbr_rowcols: int,
             image_key: str,
             image_size: int = 1000,
             extractor: str = 'efficientnet_b0_ver1'):

    """ Submits job_cnt jobs. """
    jobs = []
    # Load up an old image and resize it to desired size.
    org_img_loc = DataLocation(storage_type='s3',
                               key=image_key,
                               bucket_name=config.TEST_BUCKET)
    org_img = load_image(org_img_loc)

    img = org_img.resize((image_size, image_size)).convert("RGB")

    img_loc = DataLocation(storage_type='memory',
                           key='tmp/{}.jpg'.
                           format(str(datetime.now()).replace(' ', '_')))
    store_image(img_loc, img)

    feat_loc = DataLocation(storage_type='memory',
                            key='tmp/{}.feats.json'.
                            format(str(datetime.now()).replace(' ', '_')))
    return JobMsg(
        task_name='extract_features',
        tasks = [ExtractFeaturesMsg(
            job_token='regression_job',
            extractor=extractor,
            rowcols=[(i, i) for i in list(range(nbr_rowcols))],
            image_loc=img_loc,
            feature_loc=feat_loc
        )])


if __name__ == '__main__':
    im_size = 1000
    log = []
    for nbr_pts in [10, 50, 100, 200]:
        for name in ['efficientnet_b0_ver1', 'vgg16_coralnet_ver1']:
            job_msg = make_job(nbr_pts, im_size, name)
            res = process_job(job_msg)
            status = '{}, {}, {}: {:.2f}'.format(im_size, nbr_pts, name,
                                                 res.results[0].runtime)
            print(status)
            log.append(status)
    print(log)
