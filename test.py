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


checkpoint = torch.jit.load('/home/rehan/projects/pytorch/torch_weights/yolov8m.torchscript')
print(checkpoint)