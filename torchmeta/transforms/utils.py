from torchvision.transforms import Compose
from torchmeta.tasks import Task

def apply_wrapper(wrapper, task_or_dataset=None):
    if task_or_dataset is None:
        return wrapper

    from torchmeta.dataset import MetaDataset
    if isinstance(task_or_dataset, Task):
        return wrapper(task_or_dataset)
    elif isinstance(task_or_dataset, MetaDataset):
        if task_or_dataset.dataset_transform is None:
            dataset_transform = wrapper
        else:
            dataset_transform = Compose([
                task_or_dataset.dataset_transform, wrapper])
        task_or_dataset.dataset_transform = dataset_transform
        return task_or_dataset
    else:
        raise NotImplementedError()