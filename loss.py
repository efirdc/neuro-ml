from torch import nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = self.cosine_similarity(x, y)
        loss = 1. - loss
        loss = loss * 0.5 + 0.5
        return loss.mean()