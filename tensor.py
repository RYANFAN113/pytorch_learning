# %%
import torch
print(torch.__version__)

# %%
x = torch.empty(4, 3)
print(x)
y = torch.rand(4, 3)
print(y)
z = torch.zeros(4, 3)
print(z)
xx = torch.tensor([[5.5, 2], [4, 4], [5, 5]])
print(xx)

# %%
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.double)
print(x)
print(x.size())

# %%
#　标准均匀分布
y = torch.rand_like(x, dtype=torch.double)
#　标准正态分布
z = torch.randn_like(x, dtype=torch.double)
print(x, '\n', y)

# %%
# 加法
y = torch.rand(5, 3, dtype=torch.double)
print(x, '\n', y)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 任何使张量会发生变化的操作都有一个前缀'_'
y.add_(x)
print(y)

# %%
x = torch.rand(6, 6)
# 改变一个 tensor 的大小或者形状
y = x.view(36)
z = x.view(-1, 18)  # -1 表示由另一个维度决定
zz = x.view(-1, 9)
print(x.size(), '\n', y.size(), '\n', z.size(), '\n', zz.size())

# %%
# 使用.item()来获得元素tensor的value
print(x[3,3])
print(x[3, 3].item())

# %%
