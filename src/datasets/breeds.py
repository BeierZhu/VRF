from robustness.tools.breeds_helpers import ClassHierarchy, print_dataset_info
from robustness.tools.breeds_helpers import make_living17, make_entity30
from robustness import datasets

import os


class Entity30ID:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        location = os.path.expanduser(location)
        data_dir = os.path.join(location, "imagenet")
        info_dir = os.path.join(location, "imagenet_class_hierarchy/modified")

        hier = ClassHierarchy(info_dir)
        ret = make_entity30(info_dir, split='rand')
        superclasses, subclass_split, label_map = ret

        print_dataset_info(superclasses,
                   subclass_split,
                   label_map,
                   hier.LEAF_NUM_TO_NAME)
        train_subclasses, test_subclasses = subclass_split
        # ID
        dataset_source = datasets.CustomImageNet(data_dir, train_subclasses, transform_train=preprocess, transform_test=preprocess)
        loaders_source = dataset_source.make_loaders(num_workers, batch_size, 
                                                    shuffle_val=False)
        self.train_loader, self.test_loader = loaders_source
        self.classnames = [label_map[i] for i in range(30)]


class Entity30OOD:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        location = os.path.expanduser(location)
        data_dir = os.path.join(location, "imagenet")
        info_dir = os.path.join(location, "imagenet_class_hierarchy/modified")

        hier = ClassHierarchy(info_dir)
        ret = make_entity30(info_dir, split='rand')
        superclasses, subclass_split, label_map = ret

        print_dataset_info(superclasses,
                   subclass_split,
                   label_map,
                   hier.LEAF_NUM_TO_NAME)
        train_subclasses, test_subclasses = subclass_split
        # ID
        dataset_source = datasets.CustomImageNet(data_dir, test_subclasses, transform_train=preprocess, transform_test=preprocess)
        loaders_source = dataset_source.make_loaders(num_workers, batch_size, 
                                                    shuffle_val=False)
        self.train_loader, self.test_loader = loaders_source
        self.classnames = [label_map[i] for i in range(30)]

