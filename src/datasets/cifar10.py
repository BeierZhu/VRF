import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset, ImageFolder

cifar_classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CIFAR10:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        location = os.path.expanduser(location)

        self.train_dataset = PyTorchCIFAR10(
            root=location, download=False, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR10(
            root=location, download=False, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes

def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x

class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)

class CIFAR101:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        location = os.path.expanduser(location)
        data_root = os.path.join(location, "CIFAR-10.1")
        data = np.load(os.path.join(data_root, 'cifar10.1_v6_data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(data_root, 'cifar10.1_v6_labels.npy'), allow_pickle=True)

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.train_loader = None

        self.test_dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.classnames = cifar_classnames


class CIFAR102:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        location = os.path.expanduser(location)

        train_data = np.load(os.path.join(location, "CIFAR-10.2", 'cifar102_train.npy'), allow_pickle=True).item()
        test_data = np.load(os.path.join(location, "CIFAR-10.2", 'cifar102_test.npy'), allow_pickle=True).item()


        train_data_images = train_data['images']
        train_data_labels = train_data['labels']

        test_data_images = test_data['images']
        test_data_labels = test_data['labels']

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.test_dataset = BasicVisionDataset(
            images=test_data_images, targets=torch.Tensor(test_data_labels).long(),
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.classnames = cifar_classnames


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)


def get_class_sublist_and_mask():
    CLASS_SUBLIST = [0, 1, 2, 3, 4, 5, 7, 8, 9] # 6 is frog in CIFAR which does not in STL10
    CLASS_SUBLIST_MASK = [(i in CLASS_SUBLIST) for i in range(10)]

    return CLASS_SUBLIST, CLASS_SUBLIST_MASK

class STL10:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        location = os.path.expanduser(location)
        train_loc = os.path.join(location, 'stl10', 'train')
        test_loc = os.path.join(location, 'stl10', 'test')

        use_cuda = torch.cuda.is_available()
        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.test_dataset = ImageFolder(root=test_loc, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        
        # get sublist
        class_sublist, self.class_sublist_mask = get_class_sublist_and_mask()
        self.classnames = [cifar_classnames[i] for i in class_sublist]
    
    def project_logits(self, logits, device=None):
        if device is None:
            device = logits.device
        return project_logits(logits, self.class_sublist_mask, device)