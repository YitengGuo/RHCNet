import os.path as osp
import pickle
import shutil
import tempfile
import time
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for k, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data[0]
        name = img_metas[0]['ori_filename']

        flag = 0
        gt_bboxes = None
        try:
            gt_bboxes = data.pop('gt_bboxes')  # list[tensor[batch,num_gt,4]]
            gt_labels = data.pop('gt_labels')
            flag = 1
        except KeyError:
            pass  # 如果没有 gt_bboxes 或者 gt_labels，不做任何处理

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)

        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]

            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                #################################################
                matched_results = []
                unmatched_results = []
                undetected_gts = []

                if flag == 1 and gt_bboxes is not None:
                    # result list[num_classes, array(num_pred, 5)]
                    num_gts = gt_bboxes[i].shape[1] if gt_bboxes[i].dim() > 1 else 0
                    scores = np.concatenate(result[i], axis=0)[:, -1]
                    try:
                        score_thr = np.sort(scores)[max(scores.shape[0] - num_gts, 0)] - 1e-6
                    except:
                        score_thr = 1.0

                    for j in range(len(result[i])):
                        gt_bboxes_j = gt_bboxes[i][gt_labels[i] == j]
                        gt_bboxes_j[:, ::2] *= (img_meta['ori_shape'][1] / img_meta['img_shape'][1])
                        gt_bboxes_j[:, 1::2] *= (img_meta['ori_shape'][0] / img_meta['img_shape'][0])

                        if gt_bboxes_j.shape[0] == 0:
                            matched_results.append(torch.zeros(0, 5))
                            unmatched_results.append(torch.from_numpy(result[i][j]))
                            undetected_gts.append(torch.zeros(0, 4))
                            continue
                        elif result[i][j].shape[0] == 0:
                            matched_results.append(torch.zeros(0, 5))
                            unmatched_results.append(torch.zeros(0, 5))
                            undetected_gts.append(gt_bboxes_j)
                            continue

                        ious = bbox_overlaps(torch.from_numpy(result[i][j][:, :-1]), gt_bboxes_j)  # num_pred, num_gts
                        max_gt_ious, index = torch.max(ious, dim=0)
                        max_ious, inds = torch.max(ious, dim=1)

                        mask = torch.zeros(result[i][j].shape[0]).bool()
                        mask[index] = True
                        mask[max_ious < 0.5] = False
                        matched_results.append(torch.from_numpy(result[i][j])[mask])
                        unmatched_results.append(torch.from_numpy(result[i][j])[~mask])
                        undetected_gts.append(gt_bboxes_j[max_gt_ious < 0.5])

                    if img_meta['ori_shape'][1] > 3000:
                        thickness = 20
                        font_size = 40
                    elif img_meta['ori_shape'][1] > 1900:
                        thickness = 10
                        font_size = 20
                    else:
                        thickness = 7
                        font_size = 15

                    # 使用自定义颜色
                    img_show = model.module.show_result(
                        img_show,
                        matched_results,
                        bbox_color=(0, 255, 0),  # 绿色边框
                        text_color=(0, 255, 0),  # 绿色文字
                        thickness=thickness,
                        font_size=font_size,
                        show=False,
                        out_file=None,
                        score_thr=score_thr)

                    img_show = model.module.show_result(
                        img_show,
                        unmatched_results,
                        bbox_color=(255, 0, 0),  # 红色边框
                        text_color=(255, 255, 255),  # 白色文字
                        thickness=thickness,
                        font_size=font_size,
                        show=False,
                        out_file=None,
                        score_thr=score_thr)

                    model.module.show_result(
                        img_show,
                        undetected_gts,
                        bbox_color=(0, 0, 255),  # 蓝色边框
                        text_color=(0, 0, 255),  # 蓝色文字
                        thickness=thickness,
                        font_size=font_size,
                        show=show,
                        out_file=out_file,
                        score_thr=0.0)
                #################################################
                else:
                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank != 0:
        return None
    else:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        ordered_results = ordered_results[:size]
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    dist.barrier()

    part_list = [torch.empty_like(part_send) for _ in range(world_size)]
    dist.all_gather(part_list, part_send)

    if rank == 0:
        result_list = []
        for i in range(world_size):
            result_list.extend(pickle.loads(part_list[i].cpu().numpy().tobytes()))
        return result_list[:size]
    return None
