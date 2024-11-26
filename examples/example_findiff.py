import torch


filter1_x = torch.tensor([[1.0, -2.0, 1.0]]).T
filter1_y = torch.tensor([[1.0, -2.0, 1.0]])
print(filter1_x.shape)

filter2 = (filter1_x@filter1_y).view(1, 1, 3, 3)
filter1_x = filter1_x.view(1, 1, 1, 3)
filter1_y = filter1_y.view(1, 1, 3, 1)



# test if equivalent
x = torch.randn(1, 1, 25, 25)

y1 = torch.nn.functional.conv2d(x, filter1_x)
y2 = torch.nn.functional.conv2d(y1, filter1_y)

# convolve with filter2
y3 = torch.nn.functional.conv2d(x, filter2)


## visualize

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3)
ax[0].imshow(x[0, 0].detach().cpu().numpy())
ax[1].imshow(y2[0, 0].detach().cpu().numpy())
ax[2].imshow(y3[0, 0].detach().cpu().numpy())
plt.show()