import torch
import copy

import clip.clip as clip

from src.models import utils

# q is the estimated label distribution of pre-training dataset
# p is the label distribution counted from downstream training set
# p set to None if the downstream dataset is class-balanced
def gla_inference(zs_logits, ft_logits, q, p=None):
    if p is None:
        p = torch.ones_like(q)/zs_logits.size(1)
    logq = torch.log(q).to(zs_logits.device)
    logp = torch.log(p).to(ft_logits.device)

    return zs_logits + ft_logits - logq - logp

def wiseft_merge(zeroshot, finetuned, alpha=0.5):
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}

    # interpolate between all weights in the checkpoints
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    finetuned.load_state_dict(theta)

    return finetuned


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

    def get_logits_feats(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return inputs, outputs