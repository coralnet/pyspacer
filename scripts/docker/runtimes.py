from spacer.messages import JobReturnMsg, DataLocation, JobMsg
from spacer.messages import ExtractFeaturesMsg, DataLocation, JobMsg
from spacer.storage import store_image, load_image
from spacer import config
from datetime import datetime
from spacer.tasks import extract_features


def make_job(nbr_rowcols: int,
              image_size: int = 1000,
              extractor_name: str = 'efficientnet_b0_ver1'):

    """ Submits job_cnt jobs. """
    jobs = []
    # Load up an old image and resize it to desired size.
    org_img_loc = DataLocation(storage_type='s3',
                               key='08bfc10v7t.png',
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
    return ExtractFeaturesMsg(
            job_token='regression_job',
            feature_extractor_name=extractor_name,
            rowcols=[(i, i) for i in list(range(nbr_rowcols))],
            image_loc=img_loc,
            feature_loc=feat_loc
    )


if __name__ == '__main__':
    im_size = 1000
    log = []
    for nbr_pts in [10, 50, 100, 200]:
        for name in ['efficientnet_b0_ver1', 'vgg16_coralnet_ver1']:
            job_msg = make_job(nbr_pts, im_size, name)
            res = extract_features(job_msg)
            status = '{}, {}, {}: {:.2f}'.format(im_size, nbr_pts, name,
                                                 res.runtime)
            print(status)
            log.append(status)
    print(status)
