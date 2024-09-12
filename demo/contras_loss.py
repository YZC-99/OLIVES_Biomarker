import torch

batchsize = 64
mask = torch.eye(batchsize, dtype=torch.float32)
features = torch.randn(batchsize, 197, 128)

# 细节: torch.unbind(features, dim=1)是将features的第1维度拆分为197个张量，然后将这197个张量拼接在一起
contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #(64*197, 128)

anchor_feature = contrast_feature
anchor_dot_contrast = torch.div(torch.matual(anchor_feature,contrast_feature.T), 0.07)


print(contrast_feature.shape)