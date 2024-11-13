import torch
import torch.nn as nn


# # Example logits (model predictions) for a batch of 2 samples
# logits = torch.tensor([[[2.3, -1.2, 0.5],    # Sample 1: cat (index 0) has highest score
#                        [-0.1, 1.8, 0.6]], # Sample 2: dog (index 1) has highest score
                       
#                        [[2.3, -1.2, 0.5],    
#                        [-0.1, 1.8, 0.6]],
                       
#                        [[2.3, -1.2, 0.5],   
#                        [-0.1, 1.8, 0.6]]]) 

# # Example ground truth labels
# labels = torch.tensor([[[1, 0, 0],
#                        [0, 1, 0]],
                       
#                        [[1, 0, 0],
#                        [0, 1, 0]],
                       
#                        [[1, 0, 0],
#                        [0, 1, 0]]], dtype=torch.float32)  # Sample 1 is a cat, Sample 2 is a dog

# print(logits.shape, labels.shape)

# # Initialize CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()

# # Compute the loss
# loss = criterion(logits.view(-1, 3), labels.view(-1, 3))

# print("Computed loss:", loss)


# checkpoint = torch.jit.load('/home/rehan/projects/pytorch/torch_weights/yolov8m.torchscript')
# print(checkpoint)

# print(torch.empty(3, dtype=torch.long).random_(5))

# loss = nn.CrossEntropyLoss()
# input = torch.randn(16, 3, 5, requires_grad=True)
# target = torch.randn(16, 3, 5).softmax(dim=2)
# output = loss(input, target)
# output2 = loss(torch.squeeze(input), torch.squeeze(target))


# print(input.shape, target.shape, output.shape, output, output2.shape, output2)


N, C = 5, 4
loss = nn.NLLLoss()
# input is of size N x C x height x width
data = torch.randn(N, 16, 10, 10)
conv = nn.Conv2d(16, C, (3, 3))
m = nn.LogSoftmax(dim=3)
# each element in target has to have 0 <= value < C
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
output = loss(m(conv(data)), target)

print(m(conv(data)).shape, target.shape, output.shape, output)
