import torch.nn.functional as F

from src.models.eval import get_feats_logits_and_labels, get_logits_and_labels, compute_accuracy
from src.models.modeling import ImageClassifier, wiseft_merge
from src.args import parse_arguments
from copy import deepcopy
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

    ws = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_ose_w, best_ose_acc = 0.0, 0.0 
    best_wse_w, best_wse_acc = 0.0, 0.0

    if 'vit-32' in zeroshot_checkpoint:
        save_prefix = 'vit-32'
    else:
        save_prefix = 'vit-16'

    save_prefix_zs = f'{save_prefix}_zs'
    save_prefix_ft = f'{save_prefix}_ft'
    if 'lpft' in finetuned_checkpoint:
        save_prefix_ft = f'lpft_{save_prefix_ft}'
    if 'linear-classifier' in finetuned_checkpoint:
        save_prefix_ft = f'lp_{save_prefix_ft}'

    for dataset_name in ['ImageNet', 'ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR', 'ObjectNet']:
        print(dataset_name)
        _, zs_logits, labels, pj_fn = get_feats_logits_and_labels(zeroshot, args, save_prefix=save_prefix_zs, dataset_name=dataset_name)
        _, ft_logits, labels, pj_fn = get_feats_logits_and_labels(finetuned, args, save_prefix=save_prefix_ft, dataset_name=dataset_name)

        zs_acc = compute_accuracy(pj_fn(zs_logits), labels)
        ft_acc = compute_accuracy(pj_fn(ft_logits), labels)

        print(f" * zeroshot performance {100*zs_acc:.2f} %")
        print(f" * finetune performance {100*ft_acc:.2f} %")

        if dataset_name == 'ImageNet':
            for w in ws:
                ose_output = (1 - w) * F.softmax(pj_fn(zs_logits), dim=-1) + w * F.softmax(pj_fn(ft_logits), dim=-1)
                ose_acc = compute_accuracy(ose_output, labels)
                if ose_acc > best_ose_acc:
                    best_ose_acc = ose_acc
                    best_ose_w = w

                wiseft_model = wiseft_merge(deepcopy(zeroshot), deepcopy(finetuned), w)
                wse_logits, labels, pj_fn = get_logits_and_labels(wiseft_model, args, save_prefix=None, dataset_name=dataset_name)
                wse_acc = compute_accuracy(pj_fn(wse_logits), labels)
                if wse_acc > best_wse_acc:
                    best_wse_acc = wse_acc
                    best_wse_w = w
            
            print(f" * ose      performance {100*best_ose_acc:.2f} %")
            print(f" * wse      performance {100*best_wse_acc:.2f} %")
        else:
            ose_output = (1 - best_ose_w) * F.softmax(pj_fn(zs_logits), dim=-1) + best_ose_w * F.softmax(pj_fn(ft_logits), dim=-1)
            ose_acc = compute_accuracy(ose_output, labels)
                
            wiseft_model = wiseft_merge(deepcopy(zeroshot), deepcopy(finetuned), best_wse_w)
            wse_logits, labels, pj_fn = get_logits_and_labels(wiseft_model, args, save_prefix=None, dataset_name=dataset_name)
            wse_acc = compute_accuracy(pj_fn(wse_logits), labels)

            print(f" * ose      performance {100*ose_acc:.2f} %")
            print(f" * wse      performance {100*wse_acc:.2f} %")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
