import time
import torch
import numpy as np
from copy import deepcopy
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow, flow_to_image, np_resize_flow
from utils.misc_utils import AverageMeter, dump_results
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop
from losses.flow_loss import SelfSequenceLoss


class TrainFramework(BaseTrainer):
    def __init__(self, train_set, valid_set, model, loss_func, save_root, config):
        super(TrainFramework, self).__init__(
            train_set, valid_set, model, loss_func, save_root, config)

        self.sp_transform = RandomAffineFlow(
            self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise)
        
        if self.cfg.model == 'smurf':
            self.seq_selfsup = SelfSequenceLoss(self.cfg.ar_loss)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'l_admm', 'flow_mean', 'l_atst', 'l_ot']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        #if 'stage1' in self.cfg:
        #    if self.i_epoch == self.cfg.stage1.epoch:
        #        self.loss_func.cfg.update(self.cfg.stage1.loss)

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            
            # update configurations
            self.cfgtor(self.cfg, self.i_iter)

            # read data to device
            img1, img2 = data['img1'].to(self.rank), data['img2'].to(self.rank)
            img_pair = torch.cat([img1, img2], 1)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # zero grads
            self.optimizer.zero_grad()

            # run 1st pass
            if self.cfg.model == 'pwc':
                res_dict = self.model(img_pair, with_bk=True)
            elif self.cfg.model == 'smurf':
                res_dict = self.model(img_pair, self.cfg.train_iters, with_bk=True)
            
            flows_12, flows_21 = res_dict['flows_fw'][0], res_dict['flows_bw'][0]
            aux_12_dict, aux_21_dict = res_dict['flows_fw'][1], res_dict['flows_bw'][1]
            
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
            aux = (aux_12_dict, aux_21_dict)

            if self.cfg.run_fw:
                img1_full, img2_full = data['img1_full'].to(self.rank), data['img2_full'].to(self.rank)
                img_pair_full = torch.cat([img1_full, img2_full], 1)
                loss, l_ph, l_sm, l_admm, flow_mean, noc_ori = self.loss_func(flows, img_pair_full, aux, data['pos'])
            else:
                loss, l_ph, l_sm, l_admm, flow_mean, noc_ori = self.loss_func(flows, img_pair, aux)

            # accumulate grads from 1st pass
            scaled_loss = 1024. * loss.mean()
            scaled_loss.backward()

            if self.cfg.model == 'pwc':
                flow_ori = res_dict['flows_fw'][0][0].detach()
            elif self.cfg.model == 'smurf':
                flow_ori = res_dict['flows_fw'][0][-1].detach()

            if self.cfg.run_atst:
                img1, img2 = data['img1_ph'].to(self.rank), data['img2_ph'].to(self.rank)

                # construct augment sample
                s = {'imgs': [img1, img2], 'flows_f': [flow_ori], 'masks_f': [noc_ori]}
                st_res = self.sp_transform(deepcopy(s)) if self.cfg.run_st else deepcopy(s)
                flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]

                if not self.cfg.mask_st:
                    noc_t = torch.ones_like(noc_t)
                
                # run 2nd pass
                img_pair = torch.cat(st_res['imgs'], 1)
                
                if self.cfg.model == 'pwc':
                    flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0][0]
                    l_atst = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
                    l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

                elif self.cfg.model == 'smurf':
                    flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]
                    l_atst = self.seq_selfsup(flow_t_pred, flow_t, noc_t)

                # accumulate grads from 2nd pass
                scaled_l_atst = 1024. * self.cfg.w_ar * l_atst.mean()
                scaled_l_atst.backward()
            else:
                l_atst = torch.zeros_like(loss)

            if self.cfg.run_ot:
                img1, img2 = data['img1_ph'].to(self.rank), data['img2_ph'].to(self.rank)
                # run 3rd pass
                img_pair = torch.cat([img1, img2], 1)

                # random crop images
                img_pair, flow_t, occ_t = random_crop(img_pair, flow_ori, 1 - noc_ori, self.cfg.ot_size)

                # slic 200, random select 8~16
                if self.cfg.ot_slic:
                    img2 = img_pair[:, 3:]
                    seg_mask = run_slic_pt(img2, n_seg=200,
                                           compact=self.cfg.ot_compact, rd_select=[8, 16],
                                           fast=self.cfg.ot_fast).type_as(img2)  # Nx1xHxW
                    noise = torch.rand(img2.size()).type_as(img2)
                    img2 = img2 * (1 - seg_mask) + noise * seg_mask
                    img_pair[:, 3:] = img2

                noc_t = 1 - occ_t

                if self.cfg.model == 'pwc':
                    flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0][0]
                    l_ot = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
                    l_ot = (l_ot * noc_t).mean() / (noc_t.mean() + 1e-7)

                elif self.cfg.model == 'smurf':
                    flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]
                    l_ot = self.seq_selfsup(flow_t_pred, flow_t, noc_t)

                # accumulate grads from 3rd pass                
                scaled_l_ot = 1024. * self.cfg.w_ar * l_ot.mean()
                scaled_l_ot.backward()
            else:
                l_ot = torch.zeros_like(loss)

            # update meters
            meters = [loss.mean(), l_ph.mean(), l_sm.mean(), l_admm.mean(), flow_mean.mean(), 
                l_atst.mean(), l_ot.mean()]
            vals = [m.item() if torch.is_tensor(m) else m for m in meters]

            key_meters.update(vals, img_pair.size(0))

            # descale grads and do optimization step
            for param in [p for p in self.model.parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.rank == 0 and self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)
                self.summary_writer.add_scalar('Train_w_ar', self.cfg.w_ar, self.i_iter)
                self.summary_writer.add_scalar('Train_lr', self.scheduler.get_last_lr()[0], self.i_iter)

            if self.rank == 0 and self.i_iter % self.cfg.print_freq == 0:
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

        self.model.eval()

        end = time.time()

        error_meters = []
        all_error_names = []
        all_error_avgs = []

        if 'dump_iter' in self.cfg.keys():
            diter = self.cfg.dump_iter
        else:
            diter = -1


        n_step = 0
        for i_set, loader in enumerate(self.valid_loader):
            error_names = ['EPE', 'E_noc', 'E_occ', 'F1_all']
            if self.cfg.eval_only and "eval_iters" in self.cfg.keys() and not self.cfg.dump_en:
                error_meters.append([AverageMeter(i=len(error_names)) for _ in range(self.cfg.eval_iters)])
            else:
                error_meters.append([AverageMeter(i=len(error_names))])

            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.rank)
                gt_flows = np.concatenate([data['target']['flow'].numpy().transpose([0, 2, 3, 1]),
                                            np.ones_like(data['target']['occ'].numpy().transpose([0, 2, 3, 1])),
                                            np.around(1 - data['target']['occ'].numpy().transpose([0, 2, 3, 1]) / 255.0)],
                                            axis=3) 

                # compute output
                if self.cfg.model == 'smurf':
                    flows,aux = self.model(img_pair, iters=self.cfg.eval_iters)['flows_fw']
                    pred_flows = [flo_.detach().cpu().numpy().transpose([0, 2, 3, 1]) for flo_ in flows]

                elif self.cfg.model == 'pwc':
                    flows,aux = self.model(img_pair)['flows_fw']
                    pred_flows = [flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])]

                if self.cfg.eval_only and not self.cfg.dump_en:
                    es = [evaluate_flow(gt_flows, pflo_) for pflo_ in pred_flows]
                else: 
                    es = [evaluate_flow(gt_flows, pred_flows[diter])]
                [i_em.update([l.item() for l in es_], img_pair.size(0)) for i_em,es_ in zip(error_meters[i_set],es)]

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                gt_flows = gt_flows[:,:,:,:2]
                                
                if isinstance(pred_flows,list):
                    pred_flows = pred_flows[diter]

                if self.cfg.dump_en and i_step % self.cfg.dump_freq == 0:
                    
                    gt_flows = np_resize_flow(gt_flows, pred_flows.shape[1:3])
                    p_gt_fl = np.concatenate([flow_to_image(flo).transpose(2,0,1)[np.newaxis,:] for flo in gt_flows], axis=0)
                    p_pr_fl = np.concatenate([flow_to_image(flo).transpose(2,0,1)[np.newaxis,:] for flo in pred_flows], axis=0)
                    r_img = (img1*255).cpu().numpy().astype(np.uint8)
                    err_mag = [np.sqrt(((flo_p - flo_gt)**2).sum(axis=2)) for flo_p,flo_gt in zip(pred_flows,gt_flows)]
                    err_vec = [flow_to_image(flo_p - flo_gt) for flo_p,flo_gt in zip(pred_flows,gt_flows)]

                    dump_results(r_img.transpose(0,2,3,1).squeeze(), val_idx=i_set, samp_idx=i_step, res=es[0], samp_type="img", path=self.save_root)
                    dump_results(p_gt_fl[0].transpose(1,2,0), val_idx=i_set, samp_idx=i_step, res=es[0], samp_type="gt", path=self.save_root)
                    dump_results(p_pr_fl[0].transpose(1,2,0), val_idx=i_set, samp_idx=i_step, res=es[0], samp_type="pred", path=self.save_root)
                    dump_results((err_mag[0]*255/10).astype(np.uint8), val_idx=i_set, samp_idx=i_step, res=es[0], samp_type="err_mag", path=self.save_root)
                    dump_results(err_vec[0], val_idx=i_set, samp_idx=i_step, res=es[0], samp_type="err_vec", path=self.save_root)
                    dump_results(data['target']['occ'].numpy().transpose([0, 2, 3, 1]).squeeze().astype(np.uint8), val_idx=i_set, samp_idx=i_step, res=es[0], samp_type="occ", path=self.save_root)

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters[i_set][-1].avg)))

                if i_step > self.cfg.valid_size:
                    break
            n_step += len(loader)

            # write errors vs raft iterations
            if self.cfg.eval_only:
                for i, i_em in enumerate(error_meters[i_set]):
                    self._log.info('Test: {0} metric per iteration [{1}/{2}]\t '.format(
                        i_set, i, self.cfg.eval_iters) + ' '.join(
                        map('{:.2f}'.format, i_em.avg)))
            
            # write error to tf board.
            for value, name in zip(error_meters[i_set][-1].avg, error_names):
                self.summary_writer.add_scalar(
                    'Valid_{}_{}'.format(name, i_set), value, self.i_epoch)

            # display flows in board
            gt_flows = np_resize_flow(gt_flows, pred_flows.shape[1:3])
            p_gt_fl = np.concatenate([flow_to_image(flo).transpose(2,0,1)[np.newaxis,:] for flo in gt_flows], axis=0)
            p_pr_fl = np.concatenate([flow_to_image(flo).transpose(2,0,1)[np.newaxis,:] for flo in pred_flows], axis=0)
            p_imgs = np.concatenate([(im*255).cpu().numpy().astype(np.uint8) for im in [img1, img2]], axis=3)
            p_flows = np.concatenate([p_gt_fl, p_pr_fl], axis=3)
            im_final = np.concatenate([p_imgs, p_flows], axis=2)
            self.summary_writer.add_images('Valid_Flows_{}'.format(i_set), im_final, self.i_epoch)

            # display masks in board
            if "masks" in aux.keys():
                masks = np.concatenate([(m*255).cpu().numpy().astype(np.uint8) for m in aux["masks"]], axis=3)
                self.summary_writer.add_images('masks_{}'.format(i_set), masks, self.i_epoch)


            #all_error_avgs.extend(error_meters[i_set][-1].avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        errors_per_iter = [[i_em.avg for i_em in set_em] for set_em in zip(*error_meters)]
        score_per_iter = np.array([sum([errs[0] for errs in errs_i]) for errs_i in errors_per_iter])
        best_iter = score_per_iter.argmin()
        best_score = score_per_iter[best_iter]

        self._log.info(f'Best iteration = {best_iter}, Best score = {best_score}')
        [all_error_avgs.extend(error_meters[i_set][best_iter].avg) for i_set in range(len(self.valid_loader))]

        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0] + all_error_avgs[4], name='Sintel')

        return all_error_avgs, all_error_names
