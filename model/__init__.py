from torch import nn
def load_model(model_type: str, modified: bool = False, nc: int = 10, ) -> nn.Module:
    if "ResNet" in model_type:
        model_zoo = {
            "ResNet18": [2, 2, 2, 2],
            "ResNet34": [3, 4, 6, 3],
        }
        if modified:
            from model.resnet import TwoStageBasicBlock as BasicBlock
        else:
            from model.resnet import BasicBlock as BasicBlock
        from model.resnet import ResNet
        return ResNet(BasicBlock, model_zoo[model_type], nc)
    elif model_type == "AlexNet":
        from model.alexnet import AlexNet
        return AlexNet
    raise ValueError("Invalid model_type. Choose 'ResNet18' or 'ResNet34'.")
