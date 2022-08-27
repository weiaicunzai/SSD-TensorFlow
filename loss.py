import torch
import torch.nn as nn
import torch.nn.functional as F



class InfoNCE(nn.Module):
    def __init__(self, n_views=2, temperature=0.07):
        super().__init__()
        self.n_views = n_views
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()



    def forward(self, x):
        batch_size = x.shape[0] / 2

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(x.device)

        # Let sim(u, v) = u?v/?u??v? denote the dot product between 
        # l2 normalized u and v (i.e. cosine similarity).
        x = F.normalize(x, dim=1)

        similarity_matrix = torch.matmul(x, x.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(x.device)

        # discard diagnal
        # Given a set {˜xk} including a positive pair of 
        # examples ˜xi and ˜xj, the contrastive prediction 
        # task aims to identify ˜xj in {˜xk}k?=i for a given ˜xi.
        # i != j for constrasive loss
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # print(positives.shape)

        # The fi- nal loss is computed across all positive pairs, 
        # both (i, j) and (j, i), in a mini-batch.
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x.device)

        logits = logits / self.temperature

        loss = self.cross_entropy(logits, labels)
        return loss


#infonce_loss= InfoNCE()
#
#x = torch.Tensor(32, 128)
#
#output = infonce_loss(x)
#
#import torchvision
#from torchvision.transforms import transforms
#
#
#s = 2
#size = 100
#color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                              transforms.RandomHorizontalFlip(),
#                                              transforms.RandomApply([color_jitter], p=0.8),
#                                              transforms.RandomGrayscale(p=0.2),
#                                              transforms.GaussianBlur(kernel_size=5),
#                                              transforms.ToTensor()])
#
#dataset = torchvision.datasets.STL10('data', split='unlabeled', transform=data_transforms)
#print(dataset[10])
#fn = lambda:dataset