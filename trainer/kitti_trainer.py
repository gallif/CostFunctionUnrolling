import time
import torch
import numpy as np
from .base_trainer import BaseTrainer
from utils.flow_utils import load_flow, evaluate_flow, flow_to_image, np_resize_flow
from utils.misc_utils import AverageMeter


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'l_admm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        if 'stage1' in self.cfg:
            if self.i_epoch == self.cfg.stage1.epoch:
                self.loss_func.module.cfg.update(self.cfg.stage1.loss)

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            if self.cfg.model == "raft":
                res_dict = self.model(img_pair, iters=self.cfg.train_iters, with_bk=True)
            elif self.cfg.model == "pwc":
                res_dict = self.model(img_pair, with_bk=True)


            flows_12, flows_21 = res_dict['flows_fw'][0], res_dict['flows_bw'][0]
            aux_12_dict, aux_21_dict = res_dict['flows_fw'][1], res_dict['flows_bw'][1]
            
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            aux = (aux_12_dict, aux_21_dict)

            loss, l_ph, l_sm, l_admm, flow_mean, _ = self.loss_func(flows, img_pair, aux)

            # update meters
            meters = [loss.mean(), l_ph.mean(), l_sm.mean(), l_admm.mean(), flow_mean.mean()]
            vals = [m.item() if torch.is_tensor(m) else m for m in meters]
            key_meters.update(vals, img_pair.size(0))

            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            # loss.backward()

            scaled_loss = 1024. * loss.mean()
            scaled_loss.backward()

            for param in [p for p in self.model.parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

            if self.i_iter % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        if type(self.valid_loader) is not list:
            self.valid_loader = [self.valid_loader]

        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        n_step = 0
        for i_set, loader in enumerate(self.valid_loader):
            error_names = ['EPE', 'E_noc', 'E_occ', 'F1_all']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.device)

                res = list(map(load_flow, data['flow_occ']))
                gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
                res = list(map(load_flow, data['flow_noc']))
                _, noc_masks = [r[0] for r in res], [r[1] for r in res]

                gt_flows = [np.concatenate([flow, occ_mask, noc_mask], axis=2) for
                            flow, occ_mask, noc_mask in
                            zip(gt_flows, occ_masks, noc_masks)]

                # compute output
                if self.cfg.model == 'raft':
                    flows,aux = self.model(img_pair, iters=self.cfg.eval_iters)['flows_fw']
                    pred_flows = flows[-1].detach().cpu().numpy().transpose([0, 2, 3, 1])

                elif self.cfg.model == 'pwc':
                    flows,aux = self.model(img_pair)['flows_fw']
                    pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])

                es = evaluate_flow(gt_flows, pred_flows)
                error_meters.update([l.item() for l in es], img_pair.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters.avg)))

                if i_step > self.cfg.valid_size:
                    break
            n_step += len(loader)

            # write error to tf board.
            for value, name in zip(error_meters.avg, error_names):
                self.summary_writer.add_scalar(
                    'Valid_{}_{}'.format(name, i_set), value, self.i_epoch)

            # display flows in board
            gt_flows = [np_resize_flow(flo[np.newaxis,:,:,:2], pred_flows.shape[1:3]) for flo in gt_flows]
            p_gt_fl = np.concatenate([flow_to_image(flo.squeeze()).transpose(2,0,1)[np.newaxis,:] for flo in gt_flows], axis=0)
            p_pr_fl = np.concatenate([flow_to_image(flo.squeeze()).transpose(2,0,1)[np.newaxis,:] for flo in pred_flows], axis=0)
            p_imgs = np.concatenate([(im*255).cpu().numpy().astype(np.uint8) for im in [img1, img2]], axis=3)
            p_flows = np.concatenate([p_gt_fl, p_pr_fl], axis=3)
            self.summary_writer.add_images('Valid_Flows_{}'.format(i_set), np.concatenate([p_imgs, p_flows], axis=2), self.i_epoch)

            # display masks in board
            if "masks" in aux.keys():
                masks = np.concatenate([(m*255).cpu().numpy().astype(np.uint8) for m in aux["masks"]], axis=3)
                self.summary_writer.add_images('masks_{}'.format(i_set), masks, self.i_epoch)

            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0], name='KITTI_Flow')

        return all_error_avgs, all_error_names
