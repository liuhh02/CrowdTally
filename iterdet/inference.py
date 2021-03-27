import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.ops import RoIAlign, RoIPool

def init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_detector(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device

    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    m.use_torchvision = True
        data['img_metas'] = data['img_metas'][0].data

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result