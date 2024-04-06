def load_dataset(dataset_name):
    # Load data
    if dataset_name == "cifar10":
        nc = 10
        from dataset.cifar10 import load_data
    elif dataset_name == "cifar100":
        nc = 100
        from dataset.cifar100 import load_data
    return load_data(), nc