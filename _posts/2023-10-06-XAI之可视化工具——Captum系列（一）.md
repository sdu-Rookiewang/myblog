---

layout: post
toc: true
title: XAI之可视化工具——Captum系列（一）
date: 2023-10-06
Author: HackMagic
categories: Computer Science
tags: [XAI]
comments: true 

---

Captum是PyTorch的模型可解释性和理解库。Captum在拉丁语中的意思是解析，并包含PyTorch模型的集成梯度、显著性映射、smoothgrad、vargrad等的通用实现。它可以快速集成于深度学习特定库（如torchvision、torchtext等）构建的模型。

## 一、什么是Captum工具

随着模型复杂性的增加和由此导致的缺乏透明度，模型可解释性方法变得越来越重要。模型理解既是一个活跃的研究领域，也是使用机器学习跨行业实际应用的重点领域。Captum提供了最先进的算法，包括集成梯度，为研究人员和开发人员提供一种简单的方法来了解哪些功能有助于模型的输出。

对于模型开发人员来说，Captum可以通过促进识别有助于模型输出的不同功能来改进模型和故障排除，以便设计更好的模型并对意外模型输出进行故障排除。

Captum帮助ML研究人员更容易实现可解释性算法，该算法可以与PyTorch模型交互。Captum还允许研究人员根据图书馆中可用的其他现有算法快速将他们的工作进行基准测试。

![](/Users/administer/Documents/GitHub/myblog/images/Captum_Attribution_Algos.png)

Captum的主要受众是模型开发人员，他们希望改进他们的模型，并了解哪些功能很重要，可解释性研究人员专注于识别可以更好地解释许多类型模型的算法。

Captum也可以由在生产中使用经过培训的模型的应用工程师使用。Captum通过改进模型可解释性提供更简单的故障排除，并有可能向最终用户更好地解释他们为什么看到特定内容，例如电影推荐。

## 二、Captum的安装与使用

### 1.安装要求

- Python >= 3.6

- PyTorch >= 1.6

### 2.安装方式

最新版本的Captum可以通过`Anaconda`（更推荐）或者`pip`轻松安装。

- 与`conda`

  - Channel：`pytorch`

    ```python
    conda install captum -c pytorch
    ```

  - Channel：`conda-forge`

    ```python
    conda install captum -c conda-forge
    ```

- 与`pip`

  ```python
  pip install captum
  ```

### 3.自定义安装

如果想尝试Captum的前沿功能（并且不介意在运行代码中遇到bug），您可以直接从GitHub安装最新测试版本。安装请运行：

```python
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .
```

要自定义安装，您还可以运行上述指令的以下变形：

- `pip install -e .[insights]`：同时安装运行Captum Insights所需的所有软件包。
- `pip install -e .[dev]`：同时安装开发所需的所有工具（测试、安装、文档构建；请参阅下面的[贡献](https://github.com/pytorch/captum#contributing)）。
- `pip install -e .[tutorials]`：同时安装运行教程笔记本所需的所有软件包。

要从手动安装执行单元测试，请运行：

```python
# running a single unit test
python -m unittest -v tests.attr.test_saliency
# running all unit tests
pytest -ra
```

## 三、Captum入门指南

Captum通过探索有助于模型预测的特征，帮助您解释和理解PyTorch模型的预测。它还有助于了解哪些神经元和层对模型预测很重要。

让我们将其中一些算法应用于我们为演示目的创建的玩具模型。为了简单起见，我们将使用以下架构，但欢迎用户使用他们选择的任何PyTorch模型。

```python
import numpy as np

import torch
import torch.nn as nn

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 2)

        # initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias = nn.Parameter(torch.zeros(1,3))
        self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias = nn.Parameter(torch.ones(1,2))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))
```

让我们创建一个模型的实例，并将其设置为评估模式。

```python
model = ToyModel()
model.eval()
```

接下来，我们需要定义简单的输入和基线张量。基线属于输入空间，通常不带预测信号。零张量可以作为许多任务的基线。一些可解释性算法，如`IntegratedGradients`、`Deeplift`和`GradientShap`，旨在将输入和基线之间的变化归因于预测类或神经网络输出的值。

我们将在上述网络上应用模型可解释性算法，以了解单个神经元/层的重要性以及在最终预测中发挥重要作用的输入部分。

为了使计算具有确定性，让我们修复随机种子。

```python
torch.manual_seed(123)
np.random.seed(123)
```

让我们定义我们的输入和基线张量。基线用于一些可解释性算法，如`IntegratedGradients, DeepLift, GradientShap, NeuronConductance, LayerConductance, InternalInfluence`和`NeuronIntegratedGradients`。

```python
input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)
```

接下来，我们将使用`IntegratedGradients`算法为每个输入特征分配第一个目标输出的归因分数。

```python
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
```

输出：

```python
IG Attributions: tensor([[-0.5922, -1.5497, -1.0067],
                         [ 0.0000, -0.2219, -5.1991]])
Convergence Delta: tensor([2.3842e-07, -4.7684e-07])
```

该算法输出每个输入元素的归因分数和收敛增量。收敛三角形的绝对值越低，近似值越好。如果我们选择不返回delta，我们根本无法提供`return_convergence_delta`输入参数。返回的增量的绝对值可以解释为每个输入样本的近似误差。它也可以作为给定输入和基线的积分近似准确性的代理。如果近似误差很大，我们可以通过将`n_steps`设置为更大的值来尝试更多的积分近似步骤。并非所有算法都返回近似误差。然而，那些确实根据算法的完整性属性来计算它。

正因评分意味着该特定位置的输入对最终预测有积极贡献，而负则意味着相反。归因分数的幅度表示贡献的强度。零归因分数意味着该特定特征没有贡献。

同样，我们可以将`GradientShap`、`DeepLift`和其他归因算法应用于模型。

`GradientShap`首先从基线分布中选择随机基线，然后将std=0.09的高斯噪声添加到每个输入示例`n_samples`时间。之后，它在每个示例基线对之间选择一个随机点，并计算相对于目标类的梯度（在这种情况下，target=0）。结果归因是梯度的平均值*（输入-基线）

```python
gs = GradientShap(model)

# We define a distribution of baselines and draw `n_samples` from that
# distribution in order to estimate the expectations of gradients across all baselines
baseline_dist = torch.randn(10, 3) * 0.001
attributions, delta = gs.attribute(input, stdevs=0.09, n_samples=4, baselines=baseline_dist,
                                   target=0, return_convergence_delta=True)
print('GradientShap Attributions:', attributions)
print('Convergence Delta:', delta)
```

输出信息

```python
GradientShap Attributions: tensor([[-0.1542, -1.6229, -1.5835],
                                   [-0.3916, -0.2836, -4.6851]])
Convergence Delta: tensor([ 0.0000, -0.0005, -0.0029, -0.0084, -0.0087, -0.0405,  0.0000, -0.0084])

```

为每个`n_samples * input.shape[0]`示例计算增量。例如，用户可以平均它们：

```python
deltas_per_example = torch.mean(delta.reshape(input.shape[0], -1), dim=1)
```

为了获得每个示例的平均增量。

以下是我们如何在上述`ToyModel`上应用`DeepLift`和`DeepLiftShap`的示例。目前DeepLift的实现仅支持`Rescale`规则。有关替代实现的更多详细信息，请参阅DeepLift[论文](https://arxiv.org/abs/1704.02685)。

```python
dl = DeepLift(model)
attributions, delta = dl.attribute(input, baseline, target=0, return_convergence_delta=True)
print('DeepLift Attributions:', attributions)
print('Convergence Delta:', delta)
```

输出信息

```python
DeepLift Attributions: tensor([[-0.5922, -1.5497, -1.0067],
                               [ 0.0000, -0.2219, -5.1991])
Convergence Delta: tensor([0., 0.])
```

`DeepLift`将与`IntegratedGradients`类似的归因分数分配给输入，但它的执行时间较短。关于DeepLift，需要记住的另一件重要事情是，它目前不支持所有非线性激活类型。有关当前实施的局限性的更多详细信息，请参阅DeepLift[文件](https://arxiv.org/abs/1704.02685)。

与集成梯度类似，DeepLift为每个输入示例返回一个收敛增量得分。近似误差是收敛deltas的绝对值，可以代表算法的近似值的准确性。

现在让我们来看看`DeepLiftShap`。与`GradientShap`类似，`DeepLiftShap`使用基线分布。在下面的示例中，我们使用与`GradientShap`相同的基线分布。

```python
dl = DeepLiftShap(model)
attributions, delta = dl.attribute(input, baseline_dist, target=0, return_convergence_delta=True)
print('DeepLiftSHAP Attributions:', attributions)
print('Convergence Delta:', delta)
```

输出信息

```python
DeepLiftShap Attributions: tensor([[-5.9169e-01, -1.5491e+00, -1.0076e+00],
                                   [-4.7101e-03, -2.2300e-01, -5.1926e+00]], grad_fn=<MeanBackward1>)
Convergence Delta: tensor([-4.6120e-03, -1.6267e-03, -5.1045e-04, -1.4184e-03, -6.8886e-03,
                           -2.2224e-02,  0.0000e+00, -2.8790e-02, -4.1285e-03, -2.7295e-02,
                           -3.2349e-03, -1.6265e-03, -4.7684e-07, -1.4191e-03, -6.8889e-03,
                           -2.2224e-02,  0.0000e+00, -2.4792e-02, -4.1289e-03, -2.7296e-02])
```

`DeepLiftShap`使用`DeepLift`计算每个输入基线对的归因分数，并对所有基线的每个输入进行平均。

它计算每个输入示例基线对的增量，从而产生`input.shape[0] * baseline.shape[0]`增量值。

与GradientShap类似，为了计算基于示例的增量，我们可以按示例对它们进行平均：

```python
deltas_per_example = torch.mean(delta.reshape(input.shape[0], -1), dim=1)
```

为了平滑和提高归因的质量，我们可以通过NoiseTunnel运行`IntegratedGradients`和其他归因方法。`NoiseTunnel`允许我们使用`SmoothGrad`、`SmoothGrad_Sq`和`VarGrad`技术来通过聚合通过聚合通过添加高斯噪声产生的多个噪声样本来平滑属性。

以下是我们如何将`NoiseTunnel`与`IntegratedGradients`一起使用的示例。

```python
ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)
attributions, delta = nt.attribute(input, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,
      baselines=baseline, target=0, return_convergence_delta=True)
print('IG + SmoothGrad Attributions:', attributions)
print('Convergence Delta:', delta)
```

输出信息

```python
IG + SmoothGrad Attributions: tensor([[-0.4574, -1.5493, -1.0893],
                                      [ 0.0000, -0.2647, -5.1619]])
Convergence Delta: tensor([ 0.0000e+00,  2.3842e-07,  0.0000e+00, -2.3842e-07,  0.0000e+00,
        -4.7684e-07,  0.0000e+00, -4.7684e-07])

```

`delta`张量中的元素数量等于：`nt_samples * input.shape[0]`为了获得示例增量，我们可以平均它们：

```python
deltas_per_example = torch.mean(delta.reshape(input.shape[0], -1), dim=1)
```

让我们看看我们网络的内部，并了解哪些层和神经元对预测很重要。

我们将从`NeuronConductance`开始。`NeuronConductance`帮助我们识别对给定层中特定神经元重要的输入特征。它通过定义神经元作为输出导数相对于神经元的导数乘以神经元相对于模型输入的导数的路径积分的重要性，通过链规则分解积分的计算。

在这种情况下，我们选择分析线性层中的第一个神经元。

```python
nc = NeuronConductance(model, model.lin1)
attributions = nc.attribute(input, neuron_selector=1, target=0)
print('Neuron Attributions:', attributions)
```

输出信息

```python
Neuron Attributions: tensor([[ 0.0000,  0.0000,  0.0000],
                             [ 1.3358,  0.0000, -1.6811]])
```

层电导显示了神经元对层和给定输入的重要性。它是隐藏层路径集成梯度的扩展，也具有完整性属性。

它没有将贡献分数归因于输入特征，而是显示了所选层中每个神经元的重要性。

```python
lc = LayerConductance(model, model.lin1)
attributions, delta = lc.attribute(input, baselines=baseline, target=0, return_convergence_delta=True)
print('Layer Attributions:', attributions)
print('Convergence Delta:', delta)
```

输出

```python
Layer Attributions: tensor([[ 0.0000,  0.0000, -3.0856],
                            [ 0.0000, -0.3488, -4.9638]], grad_fn=<SumBackward1>)
Convergence Delta: tensor([0.0630, 0.1084])
```

与其他返回收敛增量的归因算法类似，`LayerConductance`返回每个示例的增量。近似误差是收敛deltas的绝对值，可以代表给定输入和基线的积分近似的准确性。

有关支持的算法列表以及如何在不同类型的模型上应用Captum的更多详细信息，请参阅我们的教程。

## 四、Captum可视化配置

### 1.网页版

Captum提供了一个名为Insights的网络界面，便于可视化并访问我们的一些可解释性算法。

通过Captum Insights在CIFAR10上分析样本模型，请运行

```python
python -m captum.insights.example
```

并导航到输出中指定的URL。

![](/Users/administer/Documents/GitHub/myblog/images/captum_insights_screenshot.png)

要构建Insights，您需要[Node ](https://nodejs.org/en/)>= 8.x和[Yarn ](https://yarnpkg.com/en/)>= 1.5。

在conda环境中从结账处构建和启动

```python
conda install -c conda-forge yarn
BUILD_INSIGHTS=1 python setup.py develop
python captum/insights/example.py
```

### 2.Jupyter版小插件

Captum Insights还有一个Jupyter小部件，提供与Web应用程序相同的用户界面。要安装和启用小部件，请运行

```python
jupyter nbextension install --py --symlink --sys-prefix captum.insights.attr_vis.widget
jupyter nbextension enable captum.insights.attr_vis.widget --py --sys-prefix
```

在conda环境中从结账构建小部件，运行

```python
conda install -c conda-forge yarn
BUILD_INSIGHTS=1 python setup.py develop
```

