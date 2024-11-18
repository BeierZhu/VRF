import os
import numpy as np
import torch

from src.models.eval import get_feats_logits_and_labels, compute_sample_accuracy
from src.models.modeling import ImageClassifier
from src.args import parse_arguments
import os
import faiss
import warnings
# mute the annoying warnings
warnings.filterwarnings("ignore", "Argument interpolation should be of type InterpolationMode instead of int")


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
    # get logit of train ImageNet
    print("ImageNet Train")
    train_zs_feats, zs_logits, labels, pj_fn = get_feats_logits_and_labels(zeroshot, 
                                                                           args, 
                                                                           save_prefix=save_prefix_zs, 
                                                                           dataset_name='ImageNet', 
                                                                           is_train=True)
    train_ft_feats, ft_logits, labels, pj_fn = get_feats_logits_and_labels(finetuned, 
                                                                           args, 
                                                                           save_prefix=save_prefix_ft, 
                                                                           dataset_name='ImageNet', 
                                                                           is_train=True)

    zs_accs = compute_sample_accuracy(pj_fn(zs_logits), labels).squeeze()
    ft_accs = compute_sample_accuracy(pj_fn(ft_logits), labels).squeeze()

    delta_index = (ft_accs & ~zs_accs).squeeze()
    delta_zs_feats = train_zs_feats[delta_index]
    delta_ft_feats = train_ft_feats[delta_index]

    ratio = delta_zs_feats.size(0) / train_zs_feats.size(0)

    delta_zs_feats /= delta_zs_feats.norm(dim=1, keepdim=True)
    delta_ft_feats /= delta_ft_feats.norm(dim=1, keepdim=True)

    delta_zs_feats = delta_zs_feats.cpu().numpy()
    delta_ft_feats = delta_ft_feats.cpu().numpy()

    index_zs = faiss.IndexFlatL2(delta_zs_feats.shape[1])
    index_zs.add(delta_zs_feats)

    index_ft = faiss.IndexFlatL2(delta_ft_feats.shape[1])
    index_ft.add(delta_ft_feats)

    K = round(ratio*1000)
    scores_dict = {}
    for dataset_name in ['ImageNet', 'ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR', 'ObjectNet']:
        print(dataset_name)
        zs_feats, _, _, _ = get_feats_logits_and_labels(zeroshot, args, save_prefix=save_prefix_zs, dataset_name=dataset_name)
        ft_feats, _, _, _ = get_feats_logits_and_labels(finetuned, args, save_prefix=save_prefix_ft, dataset_name=dataset_name)

        zs_feats /= zs_feats.norm(dim=1, keepdim=True)
        ft_feats /= ft_feats.norm(dim=1, keepdim=True)

        zs_feats = zs_feats.cpu().numpy()
        ft_feats = ft_feats.cpu().numpy()

        D_zs, _ = index_zs.search(zs_feats, K)
        D_ft, _ = index_ft.search(ft_feats, K)

        if dataset_name == 'ImageNet':
            zs_scores_in = - D_zs[:,-1]
            ft_scores_in = - D_ft[:,-1]

            num_in = zs_scores_in.shape[0]

            zs_scores_in_sorted = np.sort(zs_scores_in)
            ft_scores_in_sorted = np.sort(ft_scores_in)

            zs_thres = zs_scores_in_sorted[round(0.05 * num_in)]
            ft_thres = ft_scores_in_sorted[round(0.05 * num_in)]
            print(f"ZS thres {zs_thres}")
            print(f"FT thres {ft_thres}")
            
            scores_dict[f'zs_scores_{dataset_name}'] = zs_scores_in
            scores_dict[f'ft_scores_{dataset_name}'] = ft_scores_in
        else:
            zs_scores_ood = - D_zs[:,-1]
            ft_scores_ood = - D_ft[:,-1]
            num_out = zs_scores_ood.shape[0]

            zs_FPR = np.sum(zs_scores_ood > zs_thres) / float(num_out)
            ft_FPR = np.sum(ft_scores_ood > ft_thres) / float(num_out)
            print(f"ZS FPR {100*zs_FPR:.2f} %")
            print(f"FT FPR {100*ft_FPR:.2f} %")

            scores_dict[f'zs_scores_{dataset_name}'] = zs_scores_ood
            scores_dict[f'ft_scores_{dataset_name}'] = ft_scores_ood

    save_path = f'{save_prefix_zs}_{save_prefix_ft}_knn.npz'
    save_path = os.path.join(args.cache_dir, save_path)
    np.savez(save_path, **scores_dict)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
