import os
import json

import torch
import numpy as np
from torch.nn import functional as F

from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets
from tqdm import tqdm

def per_class_accuracy(logits, labels):
    _, predicted_classes = torch.max(logits, 1)
    num_classes = logits.size(1)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for i in range(labels.size(0)):
        label = labels[i]
        pred = predicted_classes[i]
        if pred == label:
            class_correct[label] += 1
        class_total[label] += 1
    
    class_accuracy = torch.where(class_total > 0, 100.0 * class_correct / class_total, torch.zeros_like(class_total))
    
    return class_accuracy

def compute_accuracy(logits, labels):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred)).sum().item()
    n = labels.size(0)
    acc = correct / n

    return acc

def compute_sample_accuracy(logits, labels):
    pred = logits.argmax(dim=1, keepdim=True)
    return pred.eq(labels.view_as(pred))


def get_images_wrt_distances(model, args, dataset_name, distances, t_u, t_l):
    indices_u = torch.where(distances > t_u - 1e-3)[0].tolist()
    indices_l = torch.where(distances < t_l + 1e-3)[0].tolist()

    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )


    test_set = dataset.get_test_dataset()
    if len(indices_u) == 0:
        selected_u = None
    else:
        selected_u = [(test_set[i]['images'], test_set[i]['labels']) for i in indices_u]
    
    if len(indices_l) == 0:
        selected_l = None
    else:
        selected_l = [(test_set[i]['images'], test_set[i]['labels']) for i in indices_l]

    return selected_u, selected_l


def identity(x):
    return x

    
def get_logits_and_labels(model, args, is_train=False, save_prefix=None, dataset_name="ImageNet"):
    if save_prefix is not None:
        if is_train:
            save_prefix = "train_" + save_prefix
        save_path = os.path.join(args.cache_dir, save_prefix + dataset_name + '.pt')
    device = args.device

    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    project_fn = getattr(dataset, 'project_logits', None)

    if save_prefix and os.path.exists(save_path):
        cache_data = torch.load(save_path)
        logits = cache_data['logits'].to(device)
        labels = cache_data['labels'].to(device)
    else:
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        
        model.eval()
        dataloader = get_dataloader(
            dataset, is_train=is_train, args=args, image_encoder=None)
        batched_data = enumerate(dataloader)

        logits = []
        labels = []

        with torch.no_grad():
            for i, data in tqdm(batched_data):
                data = maybe_dictionarize(data)
                x = data['images'].to(device)
                label = data['labels'].to(device)
                logit = utils.get_logits(x, model)

                labels.append(label)
                logits.append(logit)

        labels = torch.cat(labels)
        logits = torch.cat(logits)
        if save_prefix:
            torch.save({'logits': logits.cpu(), 'labels': labels.cpu()}, save_path)

    if project_fn is not None:
        return logits, labels, project_fn
    else:
        return logits, labels, identity


def get_feats_logits_and_labels(model, args, is_train=False, save_prefix=None, dataset_name="ImageNet"):
    if is_train:
        save_path = os.path.join(args.cache_dir, "train_" + save_prefix + dataset_name + '_feat.pt')
    else:
        save_path = os.path.join(args.cache_dir, save_prefix + dataset_name + '_feat.pt')
    device = args.device

    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    project_fn = getattr(dataset, 'project_logits', None)

    if os.path.exists(save_path):
        cache_data = torch.load(save_path)
        logits = cache_data['logits'].to(device)
        labels = cache_data['labels'].to(device)
        feats = cache_data['feats'].to(device)
    else:
        print(f"do not find cache in {save_path}")
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        
        model.eval()
        dataloader = get_dataloader(
            dataset, is_train=is_train, args=args, image_encoder=None)
        batched_data = enumerate(dataloader)

        logits = []
        labels = []
        feats = []

        with torch.no_grad():
            for i, data in tqdm(batched_data):
                data = maybe_dictionarize(data)
                x = data['images'].to(device)
                label = data['labels'].to(device)
                feat, logit = utils.get_feats_logits(x, model)

                labels.append(label)
                logits.append(logit)
                feats.append(feat)

        labels = torch.cat(labels)
        logits = torch.cat(logits)
        feats = torch.cat(feats)

        torch.save({'logits': logits.cpu(), 'labels': labels.cpu(), 'feats': feats.cpu()}, save_path)

    if project_fn is not None:
        return feats, logits, labels, project_fn
    else:
        return feats, logits, labels, identity