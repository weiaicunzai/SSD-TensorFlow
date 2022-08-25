import torch
import torchvision

def alexnet(num_classes, add_bn=True):
    model = torchvision.models.alexnet(pretrained=True)
    #print(model)
    if add_bn:
        features = []
        for layer in model.features:
            features.append(layer)
            if isinstance(layer, torch.nn.Conv2d):
                dim = layer.weight.shape[0]
                features.append(torch.nn.BatchNorm2d(dim))

        model.features = torch.nn.Sequential(*features)

    model.classifier[-1] = torch.nn.Linear(
        in_features=4096,
        out_features=num_classes,
        bias=True
    )
    # print(model)
    return model