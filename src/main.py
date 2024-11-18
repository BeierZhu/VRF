import os

import numpy as np

import torch

from src.models.eval import get_feats_logits_and_labels, compute_accuracy
from src.models.modeling import ImageClassifier
from src.models.calibration import Calibrater
from src.args import parse_arguments
import os
import torch.nn.functional as F


def weight(x, a=1.5, b=0.6):
    z = - (x - a) / b
    w = 1 / (1 + torch.exp(-z))
    return w

def main(args):    
    # load zeroshot and finetuned models
    zeroshot_checkpoint, finetuned_checkpoint = args.load

    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    zeroshot.process_images = True
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    zeroshot.to(args.device)
    finetuned.to(args.device)

    if 'vit-32' in zeroshot_checkpoint:
        save_prefix = 'vit-32'
    else:
        save_prefix = 'vit-16'

    save_prefix_zs = f'{save_prefix}_zs'
    save_prefix_ft = f'{save_prefix}_ft'
    if 'lpft' in finetuned_checkpoint:
        save_prefix_ft = f'lpft_{save_prefix_ft}'

    zs_feats, zs_logits, labels, pj_fn = get_feats_logits_and_labels(zeroshot, args, save_prefix=save_prefix_zs, dataset_name='ImageNet')
    ft_feats, ft_logits, labels, pj_fn = get_feats_logits_and_labels(finetuned, args, save_prefix=save_prefix_ft, dataset_name='ImageNet')

    zs_acc = compute_accuracy(pj_fn(zs_logits), labels)
    ft_acc = compute_accuracy(pj_fn(ft_logits), labels)

    zs_cali = Calibrater(zs_logits, zs_acc)
    ft_cali = Calibrater(ft_logits, ft_acc)
    T_zs, _ = zs_cali.search_T()
    T_ft, _ = ft_cali.search_T()

    load_path = f'{save_prefix_zs}_{save_prefix_ft}_knn.npz'
    load_path = os.path.join(args.cache_dir, load_path)

    data_npz = np.load(load_path)
    scores_dict =  {key: data_npz[key] for key in data_npz}

    ood_accs = []

    for dataset_name in ['ImageNet', 'ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR', 'ObjectNet']:
        zs_scores = - scores_dict[f'zs_scores_{dataset_name}']
        zs_scores = torch.tensor(zs_scores).to(args.device).unsqueeze(1)

        ft_scores = - scores_dict[f'ft_scores_{dataset_name}']
        ft_scores = torch.tensor(ft_scores).to(args.device).unsqueeze(1)  

        _, zs_logits, labels, pj_fn = get_feats_logits_and_labels(zeroshot, args, save_prefix=save_prefix_zs, dataset_name=dataset_name)
        _, ft_logits, labels, pj_fn = get_feats_logits_and_labels(finetuned, args, save_prefix=save_prefix_ft, dataset_name=dataset_name)

        zs_logits /= T_zs
        ft_logits /= T_ft

        w = weight(ft_scores) 

        zs_probs = (1 - w) * F.softmax(pj_fn(zs_logits), dim=-1)
        ft_probs = w * F.softmax(pj_fn(ft_logits), dim=-1)
        ada_acc = compute_accuracy(zs_probs + ft_probs, labels)

        if dataset_name == "ImageNet":
            id_acc = ada_acc
        
        if dataset_name in ['ImageNetV2', 'ImageNetSketch', 'ImageNetR', 'ImageNetA', 'ObjectNet']:
            ood_accs.append(ada_acc)
        
        print(f'{dataset_name}: {100*ada_acc:.2f} %')
            
    ood_acc = sum(ood_accs) / len(ood_accs)
    print(f" => ID Acc {100*id_acc:.2f} % OOD Acc {100*ood_acc:.2f} %")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
