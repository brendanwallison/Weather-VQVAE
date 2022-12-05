from dataset import DrivingDataset
import numpy as np
import cv2

# currently modifies input
def prep_data(torch_stack):
    torch_stack = torch_stack.detach().clone().permute(1, 2, 3, 0).numpy() # should 2 and 3 be swapped as well?
    torch_stack = torch_stack[:, :, :, [2, 1, 0]] # RGB to BGR
    torch_stack *= 255 # 0 to 255
    torch_stack = torch_stack.astype(np.uint8) # to int
    return torch_stack


dataset = DrivingDataset("C:/Users/brend/Downloads/geotiffs/", frames=128, skip=16)
video = dataset[8]
print(video[1].max()*255)

test = prep_data(video)

size = test.shape[1:3]
fps = 4


out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
for i in range(test.shape[0]):
    out.write(test[i])
out.release()
