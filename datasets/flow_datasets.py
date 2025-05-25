import imageio
import numpy as np
import random
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_gpu, n_frames, skip_frames=False, input_transform=None, co_transform=None, crop_transform=None,
                 target_transform=None, ap_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.skip_frames = skip_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.crop_transform = crop_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        samples = self.collect_samples()
        # verify valid length
        #N = len(samples) - len(samples) % n_gpu
        N = len(samples)

        self.samples = samples[:N]

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images = s['imgs']
        images = [imageio.imread(self.root / p).astype(np.float32) for p in images]

        target = {}
        if 'flow' in s:
            target['flow'] = sum([load_flow(self.root / f) for f in s['flow']])
        if 'mask' in s:
            # 0~255 HxWx1
            mask = imageio.imread(self.root / s['mask']).astype(np.float32) / 255.
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            target['mask'] = np.expand_dims(mask, -1)
        if 'occ' in s:
            target['occ'] = sum([imageio.imread(self.root / p).astype(np.float32) for p in s['occ']])[:,:,None]


        if self.skip_frames:
            # use only first and last for large deformations
            images = [images[0], images[-1]]
        return images, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images_raw, target = self._load_sample(self.samples[idx])
        data = {}
        # In unsupervised learning, there is no need to change target with image
        if self.co_transform is not None:
            images_full, _ = self.co_transform(images_raw, {})
        else:
            images_full = images_raw

        if self.crop_transform is not None:
            images, target = self.crop_transform(images_full, {})
            if self.input_transform is not None:
                images_full = [self.input_transform(i) for i in images_full]
            data.update({'img{}_full'.format(i + 1): p for i, p in enumerate(images_full)})
            data.update({'pos': target['pos']})
        else:
            images = images_full

        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]
        data.update({'img{}'.format(i + 1): p for i, p in enumerate(images)})

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
            for i in range(self.n_frames):
                data['img{}_ph'.format(i + 1)] = imgs_ph[i]

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = self.target_transform[key](target[key])
        data['target'] = target
        return data #, self.samples[idx]


class FlyingChairs(ImgSeqDataset):
    def __init__(self, root, n_gpu=2, n_frames=2, skip_frames=False, split='traintest',
                with_flow=False, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None,crop_transform=None ):
        self.with_flow = with_flow
        self.split = split
        root = Path(root)
        super(FlyingChairs, self).__init__(root, n_gpu, n_frames, skip_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, crop_transform=crop_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root 

        assert img_dir.isdir()
        samples = []
        if self.split == 'train' or self.split == 'test':
            split_files = open(img_dir / self.split + '.txt').readlines()
            for file in split_files:
                im1, im2, flow_dir = [img_dir / f.split('/')[-1] for f in file.strip('\n').split(',')]
                s = {'imgs': [im1, im2]}
                assert all([p.isfile() for p in s['imgs']])
                if self.with_flow:
                    s['flow'] = [flow_dir]
                    assert all([f.isfile() for f in s['flow']])
                samples.append(s)
        
        else:
            images = sorted(img_dir.glob('*.ppm'))
            flow_list = sorted(img_dir.glob('*.flo'))
            assert (len(images)//2 == len(flow_list))
        
            for i,flow_dir in enumerate(flow_list):
                im1 = images[2*i]
                im2 = images[2*i + 1]
                s = {'imgs': [im1, im2]}
                assert all([p.isfile() for p in s['imgs']])
                if self.with_flow:
                    s['flow'] = [flow_dir]
                    assert all([f.isfile() for f in s['flow']])
                samples.append(s)
        return samples


class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_gpu=2, n_frames=2, transform=None, co_transform=None, crop_transform=None):
        super(SintelRaw, self).__init__(root, n_gpu, n_frames, input_transform=transform,
                                        co_transform=co_transform, crop_transform=crop_transform)

    def collect_samples(self):
        scene_list = self.root.dirs()
        samples = []
        for scene in scene_list:
            img_list = scene.files('*.png')
            img_list.sort()

            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                sample = {'imgs': [self.root.relpathto(file) for file in seq]}
                samples.append(sample)
        return samples


class SintelTest(ImgSeqDataset):
    def __init__(self, root, n_gpu=2, n_frames=2, skip_frames=False, type='clean', split='test',
                 subsplit='trainval', with_flow=False, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None,crop_transform=None ):
        self.dataset_type = type
        self.with_flow = with_flow
        
        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(SintelTest, self).__init__(root, n_gpu, n_frames, skip_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, crop_transform=crop_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)

        assert img_dir.isdir()

        samples = []
        for img_map in sorted((self.root / img_dir).glob('*/*.png')):
            info = img_map.splitall()
            scene, filename = info[-2:]
            fid = int(filename[-8:-4])
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue

            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}
            try:
                assert all([p.isfile() for p in s['imgs']])

                if self.with_flow:
                    #if self.n_frames == 3:
                    #    # for img1 img2 img3, only flow_23 will be evaluated
                    #    s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid + 1)
                    #elif self.n_frames == 2:
                    #    # for img1 img2, flow_12 will be evaluated
                    #    s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid)
                    #else:
                    #    raise NotImplementedError(
                    #        'n_frames {} with flow or mask'.format(self.n_frames))
                    s['flow'] = [flow_dir / scene / 'frame_{:04d}.flo'.format(fid + i) for i in
                                    range(self.n_frames-1)]
                    if self.with_flow:
                        assert all([f.isfile() for f in s['flow']])
            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples


class Sintel(ImgSeqDataset):
    def __init__(self, root, n_gpu=2, n_frames=2, skip_frames=False, type='clean', split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, crop_transform=None):
        self.dataset_type = type
        self.with_flow = with_flow
        
        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split
        self.export_scene   = ['ambush_2','ambush_5']
        root = Path(root) / split
        super(Sintel, self).__init__(root, n_gpu, n_frames, skip_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, crop_transform=crop_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'
        occ_dir = self.root / 'occlusions'

        assert img_dir.isdir() and flow_dir.isdir() and occ_dir.isdir()

        samples = []
        for flow_map in sorted((self.root / flow_dir).glob('*/*.flo')):
            info = flow_map.splitall()
            scene, filename = info[-2:]
            fid = int(filename[-8:-4])
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue
                #if self.subsplit == 'val' and scene not in self.export_scene:
                #    continue


            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}
            try:
                assert all([p.isfile() for p in s['imgs']])

                if self.with_flow:
                    #if self.n_frames == 3:
                    #    # for img1 img2 img3, only flow_23 will be evaluated
                    #    s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid + 1)
                    #elif self.n_frames == 2:
                    #    # for img1 img2, flow_12 will be evaluated
                    #    s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid)
                    #else:
                    #    raise NotImplementedError(
                    #        'n_frames {} with flow or mask'.format(self.n_frames))
                    s['flow'] = [flow_dir / scene / 'frame_{:04d}.flo'.format(fid + i) for i in
                                    range(self.n_frames-1)]
                    s['occ'] = [occ_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                                    range(self.n_frames-1)]                                    
                    if self.with_flow:
                        assert all([f.isfile() for f in s['flow']]) and all([f.isfile() for f in s['occ']])
            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples


class KITTIRawFile(ImgSeqDataset):
    def __init__(self, root, sp_file, n_gpu=2, n_frames=2, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, crop_transform=None):
        self.sp_file = sp_file
        super(KITTIRawFile, self).__init__(root, n_gpu, n_frames,
                                           input_transform=transform,
                                           target_transform=target_transform,
                                           co_transform=co_transform,
                                           crop_transform=crop_transform,
                                           ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                s = {'imgs': [sp[i] for i in range(self.n_frames)]}
                samples.append(s)
            return samples


class KITTIFlowMV(ImgSeqDataset):
    """
    This dataset is used for unsupervised training only
    """

    def __init__(self, root, n_gpu=2, n_frames=2,
                 transform=None, co_transform=None, crop_transform=None, ap_transform=None, of=False ):
        self.of = of
        super(KITTIFlowMV, self).__init__(root, n_gpu, n_frames,
                                          input_transform=transform,
                                          co_transform=co_transform,
                                          crop_transform=crop_transform,
                                          ap_transform=ap_transform)

    def collect_samples(self):
        flow_occ_dir = 'flow_' + 'occ'
        assert (self.root / flow_occ_dir).isdir()

        img_l_dir, img_r_dir = 'image_2', 'image_3'
        assert (self.root / img_l_dir).isdir() and (self.root / img_r_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            for img_dir in [img_l_dir, img_r_dir]:
                img_list = (self.root / img_dir).files('*{}*.png'.format(root_filename))
                img_list.sort()

                for st in range(0, len(img_list) - self.n_frames + 1):
                    seq = img_list[st:st + self.n_frames]
                    sample = {}
                    sample['imgs'] = []
                    for i, file in enumerate(seq):
                        frame_id = int(file[-6:-4])
                        if not self.of:
                            if 12 >= frame_id >= 9:
                                break
                        else:
                            if frame_id < 10 or frame_id > 11:
                                break
                        sample['imgs'].append(self.root.relpathto(file))
                    if len(sample['imgs']) == self.n_frames:
                        samples.append(sample)
        return samples


class KITTIFlow(ImgSeqDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, n_gpu=2, n_frames=2, transform=None):
        super(KITTIFlow, self).__init__(root, n_gpu, n_frames, input_transform=transform)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # img 1 2 for 2 frames, img 0 1 2 for 3 frames.
        st = 1 if self.n_frames == 2 else 0
        ed = st + self.n_frames
        imgs = [s['img{}'.format(i)] for i in range(st, ed)]

        inputs = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]
        raw_size = inputs[0].shape[:2]

        data = {
            'flow_occ': self.root / s['flow_occ'],
            'flow_noc': self.root / s['flow_noc'],
        }

        data.update({  # for test set
            'im_shape': raw_size,
            'img1_path': self.root / s['img1'],
        })

        if self.input_transform is not None:
            inputs = [self.input_transform(i) for i in inputs]
        data.update({'img{}'.format(i + 1): inputs[i] for i in range(self.n_frames)})
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        flow_occ_dir = 'flow_' + 'occ'
        flow_noc_dir = 'flow_' + 'noc'
        assert (self.root / flow_occ_dir).isdir()

        img_dir = 'image_2'
        assert (self.root / img_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            flow_occ_map = flow_occ_dir + '/' + flow_map
            flow_noc_map = flow_noc_dir + '/' + flow_map
            s = {'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map}

            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).isfile() and (self.root / img2).isfile()
            s.update({'img1': img1, 'img2': img2})
            if self.n_frames == 3:
                img0 = img_dir + '/' + root_filename + '_09.png'
                assert (self.root / img0).isfile()
                s.update({'img0': img0})
            samples.append(s)
        return samples

class KITTIFlowTest(ImgSeqDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, n_gpu=2, n_frames=2, transform=None, type='kitti15'):
        self.dataset_type = type
        super(KITTIFlowTest, self).__init__(root, n_gpu, n_frames, input_transform=transform)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # img 1 2 for 2 frames, img 0 1 2 for 3 frames.
        st = 1 if self.n_frames == 2 else 0
        ed = st + self.n_frames
        imgs = [s['img{}'.format(i)] for i in range(st, ed)]

        inputs = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]
        raw_size = inputs[0].shape[:2]

        data = {}
        #data = {
        #    'flow_occ': self.root / s['flow_occ'],
        #    'flow_noc': self.root / s['flow_noc'],
        #}

        data.update({  # for test set
            'im_shape': raw_size,
            'img1_path': self.root / s['img1'],
        })

        if self.input_transform is not None:
            inputs = [self.input_transform(i) for i in inputs]
        data.update({'img{}'.format(i + 1): inputs[i] for i in range(self.n_frames)})
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        if self.dataset_type == 'kitti15':
            img_dir = 'image_2'
        elif self.dataset_type == 'kitti12':
            img_dir = 'colored_0'
        assert (self.root / img_dir).isdir()

        samples = []
        for img_map in sorted((self.root / img_dir).glob('*10.png')):
            img_map = img_map.basename()
            root_filename = img_map[:-7]

            s = {}

            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).isfile() and (self.root / img2).isfile()
            s.update({'img1': img1, 'img2': img2})
            if self.n_frames == 3:
                img0 = img_dir + '/' + root_filename + '_09.png'
                assert (self.root / img0).isfile()
                s.update({'img0': img0})
            samples.append(s)
        return samples

