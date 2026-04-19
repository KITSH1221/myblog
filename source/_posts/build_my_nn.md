---
title: build_my_nn
date: 2026-04-18 15:40:00
tags:
---
# 前向传播
l: 神经网络的层编号

j: 当前层（l 层）中神经元的索引

i: 前一层（l-1 层）中神经元的索引

$n_{l-1}$: 前一层（l-1 层）的神经元总数

$z_j^{[l]}$: 第 l 层第 j 个神经元的线性输出（或称“预激活值”）

$a_j^{[l]}$: 第 l 层第 j 个神经元的激活输出（即该神经元的最终输出）

$w_{ji}^{[l]}$: 连接前一层第 i 个神经元与当前层第 j 个神经元的权重

$$a^{\left [ l\right ]}_{j}=g^{\left [ l\right ]}(z^{\left [ l\right ]}_{j})$$

$$z^{\left [ l\right ]}_{j}=\sum_{i=0}^{n_{l-1}}{(w^{\left [ l\right ]}_{ji}\cdot{a_{i}^{\left [ l-1\right ]}})}$$

<br>

# 反向传播

$$pred=w^{T}x$$

$$ e=\frac{1}{2}(pred-goal)^{2} $$

$$\frac{\mathrm{d} e}{\mathrm{d} w}=(pred-goal)\frac{\mathrm{d} pred}{\mathrm{d} w}=(pred-goal)x $$

$$w=w-\alpha\frac{\mathrm{d} e}{\mathrm{d} w}$$


``` python
#numpy implementation of a simple 3-layer neural network with ReLU activation
from os import error
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

relu = lambda x: x * (x > 0)
relu_deriv = lambda x: 1. * (x > 0)

alpha=0.2
hidden_size=4

weight_0_1=2*np.random.random((3, hidden_size))-1
weight_1_2=2*np.random.random((hidden_size,1))-1

input=np.array([[1,0,1],
                [0,1,1],
                [0,0,1],
                [1,1,1]])
expected_output=np.array([[1,1,0,0]]).T

error_list=[]
for iteration in range(60):
    error_layer_2=0
    for i in range(len(input)):
        #forward
        layer_0 = input[i:i+1]
        layer_1 = relu(layer_0.dot(weight_0_1))
        layer_2 = layer_1.dot(weight_1_2)

        error_layer_2+=np.sum((layer_2-expected_output[i:i+1])**2)
        #calculate deltas and backpropagate
        layer_2_delta = layer_2 - expected_output[i:i+1]
        layer_1_delta = layer_2_delta.dot(weight_1_2.T) * relu_deriv(layer_1)
        
        weight_1_2 -= alpha * layer_1.T @ layer_2_delta
        weight_0_1 -= alpha * layer_0.T @ layer_1_delta
    if iteration % 9 == 0:
        print(f"Error: {error_layer_2}")
        error_list.append(error_layer_2)
plt.plot(error_list)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error over Iterations')
plt.show()
```

error:3.8997781142999073<br>
error:1.7731885718099<br>
error:1.1286997023958958<br>
error:0.449918434891837<br>
error:0.16076246800829808<br>
error:0.05905597001514479<br>
error:0.01568483838391143<br>
error:0.0039657376257124155<br>
error:0.0007521607873005097<br>
error:0.00010739002363998628<br>
error:2.424871981987816e-05<br>
error:4.399288354160487e-06<br>

![Error over Iteration](/image/output.png)


# Base on pytorch

``` python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
```