import copy
from torchvision import transforms

from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms, get_crop_transform
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms import sep_transforms
from datasets.flow_datasets import FlyingChairs
from datasets.flow_datasets import SintelRaw, Sintel, SintelTest
from datasets.flow_datasets import KITTIRawFile, KITTIFlow, KITTIFlowMV, KITTIFlowTest


def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)
    crop_transform = get_crop_transform(aug_args=all_cfg.data_aug)

    if cfg.type == 'Sintel_Flow':
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

        train_set_1 = Sintel(cfg.root_sintel, n_gpu=cfg.n_gpu, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=False,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             crop_transform=crop_transform
                             )
        train_set_2 = Sintel(cfg.root_sintel, n_gpu=cfg.n_gpu, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=False,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             crop_transform=crop_transform
                             )
        train_set = ConcatDataset([train_set_1, train_set_2])

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean',
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor(),
                                                'occ': sep_transforms.ArrayToTensor()}
                             )
        valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final',
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor(),
                                                'occ': sep_transforms.ArrayToTensor()}
                             )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    elif cfg.type == 'Sintel_Test':
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

        train_set_1 = Sintel(cfg.root_sintel, n_gpu=cfg.n_gpu, n_frames=cfg.train_n_frames, type='clean',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=False,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             crop_transform=crop_transform
                             )
        train_set_2 = Sintel(cfg.root_sintel, n_gpu=cfg.n_gpu, n_frames=cfg.train_n_frames, type='final',
                             split='training', subsplit=cfg.train_subsplit,
                             with_flow=False,
                             ap_transform=ap_transform,
                             transform=input_transform,
                             co_transform=co_transform,
                             crop_transform=crop_transform
                             )
        train_set = ConcatDataset([train_set_1, train_set_2])

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = SintelTest(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean',
                             split=cfg.val_split, with_flow=False, transform=valid_input_transform)
        valid_set_2 = SintelTest(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final',
                             split=cfg.val_split, with_flow=False, transform=valid_input_transform)
        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    
    elif cfg.type == 'Sintel_Raw':
        train_set = SintelRaw(cfg.root_sintel_raw, n_gpu=cfg.n_gpu, n_frames=cfg.train_n_frames,
                              transform=input_transform, co_transform=co_transform, crop_transform=crop_transform)
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        valid_set_1 = Sintel(cfg.root_sintel, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames, skip_frames=cfg.s_frames, type='clean',
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor(),
                                                'occ': sep_transforms.ArrayToTensor()}
                            )
        valid_set_2 = Sintel(cfg.root_sintel, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames, skip_frames=cfg.s_frames, type='final',
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor(),
                                                'occ': sep_transforms.ArrayToTensor()}
                            )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
    
    elif cfg.type == 'KITTI_Raw':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIRawFile(
            cfg.root,
            cfg.train_file,
            n_gpu=cfg.n_gpu,
            n_frames=cfg.train_n_frames,
            transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            crop_transform=crop_transform  # no target here
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform,
                                )
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform,
                                )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    elif cfg.type == 'FlyingChairs':
        train_input_transform = copy.deepcopy(input_transform)
        #train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = FlyingChairs(
            cfg.root,
            n_gpu=cfg.n_gpu,
            split='train',
            transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            crop_transform=crop_transform  # no target here
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set = FlyingChairs(
            cfg.root, 
            n_gpu=cfg.n_gpu, 
            split='test',
            with_flow=True,
            transform=valid_input_transform,
        )
    
    elif cfg.type == 'KITTI_MV':
        train_input_transform = copy.deepcopy(input_transform)
        #train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIFlowMV(
            root_flow,
            n_gpu=cfg.n_gpu,
            n_frames=cfg.train_n_frames,
            transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            crop_transform=crop_transform  # no target here
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform,
                                )
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform,
                                )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
    
    elif cfg.type == 'KITTI_OF':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIFlowMV(
            root_flow,
            n_gpu=cfg.n_gpu,
            n_frames=cfg.train_n_frames,
            transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            crop_transform=crop_transform,  # no target here
            of=True
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform,
                                )
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform,
                                )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
    
    elif cfg.type == 'Kitti15_Test':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIFlowMV(
            root_flow,
            n_gpu=cfg.n_gpu,
            n_frames=cfg.train_n_frames,
            transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
            crop_transform=crop_transform  # no target here
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlowTest(cfg.root_kitti15.replace('training','testing'), n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform, type='kitti15'
                                )
        valid_set_2 = KITTIFlowTest(cfg.root_kitti12.replace('training','testing'), n_gpu=cfg.n_gpu, n_frames=cfg.val_n_frames,
                                transform=valid_input_transform, type='kitti12'
                                )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    else:
        raise NotImplementedError(cfg.type)
    return train_set, valid_set