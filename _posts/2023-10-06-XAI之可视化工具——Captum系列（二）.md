---
layout: post
toc: true
title: XAI之可视化工具——Captum系列（二）
date: 2023-10-06
Author: HackMagic
categories: Computer Science
tags: [XAI]
comments: true 
---



## 前言

```python
import torch
import torch.nn.functional as F

from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

import os
import json
```



## 一、LIME检查图像分类

在本节中，我们将学习应用Lime来分析在ImageNet-1k上训练的Resnet。对于测试数据，我们使用PASCAL VOC 2012的样本，因为它的分割掩码可以直接作为图像的语义“超级像素”。

```python
from torchvision.models import resnet18
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from captum.attr._core.lime import get_exp_kernel_similarity_function

from PIL import Image
import matplotlib.pyplot as plt
```

### 1.加载模型与数据集

我们可以直接从torchvision加载预训练的Resnet，并将其设置为评估模式，作为我们的目标图像分类器进行检查。

```python
resnet = resnet18(pretrained=True)
resnet = resnet.eval()
```

该模型预测给定示例图像的ImageNet-1k标签。为了更好地呈现结果，我们还加载了标签索引和文本的映射。

```python
!wget -P $HOME/.torch/models https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.
```

```python
labels_path = os.getenv('HOME') + '/.torch/models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = {idx: label for idx, [_, label] in json.load(json_data).items()}
```

如前所述，我们将使用PASCAL VOC 2012作为测试数据，该数据在torchvision中也可用。我们将用`torchvision`变换加载它，该变换将图像和目标（即分割掩码）转换为张量。

```python
voc_ds = VOCSegmentation(
    './VOC',
    year='2012',
    image_set='train',
    download=False,
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )    
    ]),
    target_transform=T.Lambda(
        lambda p: torch.tensor(p.getdata()).view(1, p.size[1], p.size[0])
    )
)
```

此数据集与每个图像一起提供一个附加分割掩码。与检查每个像素相比，片段（或“超级像素”）在语义上对人类来说更直观。我们将在第1.3节中讨论更多。

让我们选择一个例子，看看图像和相应的面具是什么样子的。在这里，我们选择除背景外还有多个片段的图像，以便我们可以比较每个片段对分类的影响。

```python
sample_idx = 439

def show_image(ind): 
    fig, ax = plt.subplots(1, 2, figsize=[6.4 * 2, 4.8])
    for i, (name, source) in enumerate(zip(['Image', 'Mask'], [voc_ds.images, voc_ds.masks])):
        ax[i].imshow(Image.open(source[ind]));
        ax[i].set_title(f"{name} {ind}")
        ax[i].axis('off')

show_image(sample_idx)
```



### 2.基线分类

我们可以检查我们的模型与上述示例的效果如何。原始的Resnet只提供标签的logits，因此我们将添加一个softmax层，将它们归一化为概率。

```python
sample_idx = 439

def show_image(ind): 
    fig, ax = plt.subplots(1, 2, figsize=[6.4 * 2, 4.8])
    for i, (name, source) in enumerate(zip(['Image', 'Mask'], [voc_ds.images, voc_ds.masks])):
        ax[i].imshow(Image.open(source[ind]));
        ax[i].set_title(f"{name} {ind}")
        ax[i].axis('off')

show_image(sample_idx)
```

然后，我们展示前5个预测标签来验证结果。

```python
sample_idx = 439

def show_image(ind): 
    fig, ax = plt.subplots(1, 2, figsize=[6.4 * 2, 4.8])
    for i, (name, source) in enumerate(zip(['Image', 'Mask'], [voc_ds.images, voc_ds.masks])):
        ax[i].imshow(Image.open(source[ind]));
        ax[i].set_title(f"{name} {ind}")
        ax[i].axis('off')

show_image(sample_idx)
```

正如我们所看到的，结果相当合理。

### 3.用LIME检查模型预测

在本节中，我们将从Captum引入LIME，以分析Resnet如何根据示例图像进行上述预测。

像许多其他Captum算法一样，Lime还支持作为一个组一起分析一些输入特征。这在处理图像时非常有用，其中每个像素中的每个颜色通道都是一个输入功能。这种群体也被称为“超级像素”。为了在输入特征上定义我们所需的组，我们只需要提供一个特征掩码。

在图像输入的情况下，特征蒙版是相同大小的2D图像，其中蒙版中的每个像素通过整数值表示其所属的特征组。相同值的像素定义了一个组。

这意味着我们可以很容易地使用VOC的分割掩码作为Captum的特征掩码！然而，虽然分段数从0到255不等，但为了提高效率，Captum更喜欢连续的组ID。因此，我们还将包括转换口罩ID的额外步骤。

```python
seg_ids = sorted(seg_mask.unique().tolist())
print('Segmentation IDs:', seg_ids)

# map segment IDs to feature group IDs
feature_mask = seg_mask.clone()
for i, seg_id in enumerate(seg_ids):
    feature_mask[feature_mask == seg_id] = i
    
print('Feature mask IDs:', feature_mask.unique().tolist())
```

现在是时候配置我们的Lime算法了。从本质上讲，Lime训练了一个可解释的代理模型来模拟目标模型的预测。因此，建立一个适当的可解释模型是Lime中最关键的一步。幸运的是，Captum提供了许多最常见的可解释模型来节省精力。我们将演示线性回归和线性套索的用法。另一个重要因素是相似性函数。由于Lime旨在解释一个示例的局部行为，它将根据训练样本的相似距离重新加权。默认情况下，Captum的Lime在conzine距离之上使用指数内核。我们将改为欧几里得距离，这在视觉上更受欢迎。

```python
exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

lr_lime = Lime(
    resnet, 
    interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
    similarity_func=exp_eucl_distance
)
```

接下来，我们将分析这些群体对最自信的预测`television`的影响。每次我们调用Lime的`attribute`函数时，都会围绕给定的输入训练一个可解释的模型，因此与许多其他Captum的归因算法不同，强烈建议只提供单个示例作为输入（具有第一维度或批处理大小的张量=1的张量）。有传递批处理输入的高级用例。感兴趣的读者可以查看[文档](https://captum.ai/api/lime.html)以了解详细信息。

为了训练可解释的模型，我们需要通过参数`n_samples`指定足够的训练数据。Lime以可解释表示的形式创建扰动样本，即表示特征“存在”或“缺失”的二进制向量。Lime需要继续调用目标模型，以获取所有扰动样品的标签/值。这个过程可能非常耗时，这取决于目标模型的复杂性和样本数量。设置`perturbations_per_eval`可以在一个正向通道中批处理多个样本，只要您的机器仍有容量，就可以缩短流程。您还可以考虑打开标志`show_progress`，以显示显示剩余转发呼叫数量的进度条。

```python
label_idx = output_probs.argmax().unsqueeze(0)
attrs = lr_lime.attribute(
    img.unsqueeze(0),
    target=label_idx,
    feature_mask=feature_mask.unsqueeze(0),
    n_samples=40,
    perturbations_per_eval=16,
    show_progress=True
).squeeze(0)

print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
```

现在，让我们使用Captum的可视化工具来查看归因热图。

```python
def show_attr(attr_map):
    viz.visualize_image_attr(
        attr_map.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels 
        method='heat_map',
        sign='all',
        show_colorbar=True
    )
    
show_attr(attrs)
```

结果看起来不错：电视部分确实与预测表现出最强的正相关性，而椅子的影响相对较小，边框略有负面贡献。

然而，我们可以进一步改善这一结果。可解释性的一个理想特征是人类易于理解。我们应该帮助减少嘈杂的干扰，并强调真正有影响力的特征。在我们的案例中，所有功能或多或少都显示出一些影响。在可解释的模型中添加套索正则化可以有效地帮助我们过滤它们。因此，让我们尝试具有拟合系数`alpha`的线性套索。对于所有内置的sklearn包装模型，您可以直接传递任何sklearn支持的参数。

此外，由于我们的示例只有4个部分，因此总共只有16种可解释表示的可能组合。因此，我们可以用尽它们，而不是随机抽样。`Lime`类的参数`perturb_func`允许我们传递一个生成器函数产生样本。我们将创建迭代组合的生成器函数，并将`n_samples`设置为其确切的长度。

```python
n_interpret_features = len(seg_ids)

def iter_combinations(*args, **kwargs):
    for i in range(2 ** n_interpret_features):
        yield torch.tensor([int(d) for d in bin(i)[2:].zfill(n_interpret_features)]).unsqueeze(0)
    
lasso_lime = Lime(
    resnet, 
    interpretable_model=SkLearnLasso(alpha=0.08),
    similarity_func=exp_eucl_distance,
    perturb_func=iter_combinations
)

attrs = lasso_lime.attribute(
    img.unsqueeze(0),
    target=label_idx,
    feature_mask=feature_mask.unsqueeze(0),
    n_samples=2 ** n_interpret_features,
    perturbations_per_eval=16,
    show_progress=True
).squeeze(0)

print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
show_attr(attrs)
```

正如我们所看到的，新的归因结果在Lasso的帮助下移除了椅子和边框。

另一个需要探索的有趣问题是，模型是否也能识别图像中的椅子。为了回答这个问题，我们将使用ImageNet中最相关的标签`rocking_chair`作为目标，其标签索引为`765`。我们可以检查模型对替代对象的信心程度。

```python
alter_label_idx = 765

alter_prob = output_probs[alter_label_idx].item()
print(f'{idx_to_labels[str(alter_label_idx)]} ({alter_label_idx}):', round(alter_prob, 4))
```

然后，我们将用我们的Lasso Lime重做归因。

```python
attrs = lasso_lime.attribute(
    img.unsqueeze(0),
    target=alter_label_idx,
    feature_mask=feature_mask.unsqueeze(0),
    n_samples=2 ** n_interpret_features,
    perturbations_per_eval=16,
    show_progress=True,
    return_input_shape=True,
).squeeze(0)

print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
show_attr(attrs)
```

如热图所示，我们的ResNet确实对椅子部分提出了正确的信念。然而，它受到前景电视部分的阻碍。这也可以解释为什么模特对椅子不如电视有信心。

### 4.了解采样过程

我们已经学会了如何使用Captum's Lime。本节还将深入研究内部采样过程，让感兴趣的读者了解下面发生的事情。采样过程的目标是为代理模型收集一组训练数据。每个数据点由三部分组成：可解释的输入、模型预测标签和相似性权重。我们将大致说明Lime如何在幕后实现他们每个人。

正如我们之前提到的，石灰从可解释的空间中采样数据。默认情况下，Lime使用给定掩码组的预设或缺席作为可解释的特征。在我们的示例中，面对上述4个段的图像，可解释的表示是4个值的二进制向量，指示每个段是否存在。这就是为什么我们知道只有16种可能的解释表示，并且可以用我们的`iter_combinations`来用尽它们。Lime将继续调用其`perturb_func`，以获取样本可解释的输入。让我们模拟这个步骤，并给我们一个这样的可解释的输入。

```python
SAMPLE_INDEX = 13

pertubed_genertator = iter_combinations()
for _ in range(SAMPLE_INDEX + 1):
    sample_interp_inp = next(pertubed_genertator)
    
print('Perturbed interpretable sample:', sample_interp_inp)
```

我们的输入样本`[1, 1, 0, 1]`意味着第三段（电视）不存在，而其他三个段停留。

为了找出目标ImageNet对该样本的预测是什么，Lime需要将其从可解释空间转换回原始示例空间，即图像空间。转换采用原始示例输入，并通过将缺席组的特征设置为默认为`0`的基线值来修改它。转换函数在Lime下称为`from_interp_rep_transform`。我们将在这里手动运行它，以获取pertubed图像输入，然后可视化它的样子。

```python
pertubed_img = lasso_lime.from_interp_rep_transform(
    sample_interp_inp,
    img.unsqueeze(0),
    feature_mask=feature_mask.unsqueeze(0),
    baselines=0
)

# invert the normalization for render
invert_norm = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

plt.imshow(invert_norm(pertubed_img).squeeze(0).permute(1, 2, 0).numpy())
plt.axis('off')
plt.show()
```

如上所示，与原始图像相比，缺席的特征，即电视部分，在扰动的图像中被掩盖，而其他当前特征保持不变。通过扰动的图像，Lime能够找到模型的预测。让我们仍然使用“电视”作为我们的归因目标，因此扰动样本的标签是模型在“电视”上预测的值。出于好奇，我们还可以检查模型的预测如何随着扰动而变化。

```python
perturbed_outputs = resnet(pertubed_img).squeeze(0).detach()
sample_label = perturbed_outputs[label_idx.item()]
print('Label of the perturbed sample as Television:', sample_label)

print('\nProbabilities of the perturbed image')
perturbed_output_probs = F.softmax(perturbed_outputs, dim=0)
print_result(perturbed_output_probs, topk=5)
print(f'\ntelevision ({label_idx.item()}):', perturbed_output_probs[label_idx].item())
```

合理的是，我们的ImageNet不再对将图像归类为电视有信心。

最后，由于Lime专注于局部可解释性，它将计算扰动图像和原始图像之间的相似性，以重新权衡该数据点的损失。请注意，计算基于输入空间而不是可解释空间。这一步只是将两个图像张量传递到给定的similarity_func参数中，在我们的案例中，这是欧几里得距离的指数内核。

```python
sample_similarity = exp_eucl_distance(img.unsqueeze(0), pertubed_img, None)
print('Sample similarity:', sample_similarity)
```

这基本上就是Lime如何创建`sample_interp_inp`、`sample_label`和`sample_similarity`的单个训练数据点。通过重复此过程`n_samples`次数，它收集了一个数据集来训练可解释的模型。

值得注意的是，我们在本节中展示的步骤是基于我们上面配置的Lime实例的示例。每个步骤的逻辑都可以自定义，特别是LimeBase类，该类将在第2节中演示。

## 二、LIME检查文本分类

在本节中，我们将使用新闻主题分类示例来演示Lime中更多可定制的功能。我们将在AG_NEWS数据集上训练一个简单的嵌入袋分类器，并分析其对单词的理解。

```python
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from collections import Counter

from IPython.core.display import HTML, display
```

### 1.加载数据并定义模型

`torchtext`包括AG_NEWS数据集，但由于它只拆分为火车和测试，我们需要从原始火车拆分中进一步削减验证集。然后，我们根据我们的火车分裂来建立常用单词的词汇量。

```python
ag_ds = list(AG_NEWS(split='train'))

ag_train, ag_val = ag_ds[:100000], ag_ds[100000:]

tokenizer = get_tokenizer('basic_english')
word_counter = Counter()
for (label, line) in ag_train:
    word_counter.update(tokenizer(line))
voc = Vocab(word_counter, min_freq=10)

print('Vocabulary size:', len(voc))

num_class = len(set(label for label, _ in ag_train))
print('Num of classes:', num_class)
```

我们使用的模型由一个嵌入袋（将单词嵌入作为潜在文本表示）和最后一个线性层（将潜在向量映射到logits）组成。顺便说一句，`pytorch`的嵌入袋并不假设第一个维度是批处理的。相反，它需要一个带有额外偏移张量的扁平化索引向量来标记每个示例的起始位置。有关详细信息，您可以参考其[文档](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#embeddingbag)。

```python
class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, num_class)

    def forward(self, inputs, offsets):
        embedded = self.embedding(inputs, offsets)
        return self.linear(embedded)
```

### 2.训练和基线分类

为了训练我们的分类器，我们需要定义一个整理函数，将样本批处理到嵌入袋所需的张量部分中，并创建可互通的数据加载器。

```python
BATCH_SIZE = 64

def collate_batch(batch):
    labels = torch.tensor([label - 1 for label, _ in batch]) 
    text_list = [tokenizer(line) for _, line in batch]
    
    # flatten tokens across the whole batch
    text = torch.tensor([voc[t] for tokens in text_list for t in tokens])
    # the offset of each example
    offsets = torch.tensor(
        [0] + [len(tokens) for tokens in text_list][:-1]
    ).cumsum(dim=0)

    return labels, text, offsets

train_loader = DataLoader(ag_train, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(ag_val, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_batch)
```

然后，我们将使用常见的交叉熵损耗和Adam优化器来训练我们的嵌入袋模型。由于这项任务的简单性，5个纪元应该足以让我们获得90%的稳定验证准确性。

```python
EPOCHS = 7
EMB_SIZE = 64
CHECKPOINT = './models/embedding_bag_ag_news.pt'
USE_PRETRAINED = True  # change to False if you want to retrain your own model

def train_model(train_loader, val_loader):
    model = EmbeddingBagModel(len(voc), EMB_SIZE, num_class)
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(1, EPOCHS + 1):      
        # training
        model.train()
        total_acc, total_count = 0, 0
        
        for idx, (label, text, offsets) in enumerate(train_loader):
            optimizer.zero_grad()
            predited_label = model(text, offsets)
            loss(predited_label, label).backward()
            optimizer.step()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

            if (idx + 1) % 500 == 0:
                print('epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(
                    epoch, idx + 1, len(train_loader), total_acc / total_count
                ))
                total_acc, total_count = 0, 0       
        
        # evaluation
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for label, text, offsets in val_loader:
                predited_label = model(text, offsets)
                total_acc += (predited_label.argmax(1) == label).sum().item()
                total_count += label.size(0)

        print('-' * 59)
        print('end of epoch {:3d} | valid accuracy {:8.3f} '.format(epoch, total_acc / total_count))
        print('-' * 59)
    
    torch.save(model, CHECKPOINT)
    return model
        
eb_model = torch.load(CHECKPOINT) if USE_PRETRAINED else train_model(train_loader, val_loader)
```

现在，让我们了解以下体育新闻，测试一下我们的模型表现如何。

```python
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
            '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
            'And it is too early to tell if he will match Aleksandr Dityatin, '
            'the Soviet gymnast who won eight total medals in 1980.')

test_labels, test_text, test_offsets = collate_batch([(test_label, test_line)])

probs = F.softmax(eb_model(test_text, test_offsets), dim=1).squeeze(0)
print('Prediction probability:', round(probs[test_labels[0]].item(), 4))
```

我们的嵌入袋确实以相当高的信心成功地将上述新闻识别为体育。

### 3.使用LIME检查模型预测

最后，是时候带回Lime来检查模型是如何进行预测的了。然而，这次我们将使用更可定制的`LimeBase`类，这也是为我们之前使用的`Lime`类提供动力的低级实现。在从扰动的二进制可解释表示中创建特征时，`Lime`类是固执己见的。它只能将“absense”功能设置为一些基线值，同时保留其他“presense”功能。在这种情况下，这不是我们想要的。对于文本，可解释的表示是一个二进制向量，指示每个位置的单词是否存在。相应的文本输入应该从字面上删除缺失的单词，这样我们的嵌入袋就可以计算左单词的平均嵌入。将它们设置为任何基线都会污染计算，此外，我们的嵌入袋根本没有像`<padding>`常见基线令牌。因此，我们必须使用`LimeBase`通过`from_interp_rep_transform`参数自定义转换逻辑。

`LimeBase`一点也不固执己见，所以我们必须手动定义每件作品。让我们按顺序讨论它们：

- `forward_func`，模型的正向函数。请注意，我们不能直接通过我们的模型，因为Captum总是假设第一个维度是批次的，而我们的嵌入袋需要扁平化的索引。因此，我们将稍后在调用`attribute`时添加虚拟维度，并在给出模型之前在这里制作一个包装来删除虚拟维度。
- `interpretable_model`，代理模型。这与我们在上述图像分类示例中演示的效果相同。我们也在这里使用sklearn线性套索。
- `similarity_func`，计算训练样本重量的函数。用于文本的最常见距离是其潜在嵌入空间中的余弦相似性。文本输入只是令牌索引的序列，因此我们必须利用模型中经过训练的嵌入层将它们编码到其潜在向量。由于这个额外的编码步骤，我们无法像图像分类示例中那样使用util `get_exp_kernel_similarity_function('cosine')`直接计算给定输入的余弦相似性。
- `perturb_func`，对可解释的表示进行采样的功能。除了使用上述图像分类示例所示的生成器外，我们还提供了另一种定义此参数的方法。在这里，我们直接定义一个函数，每次调用都返回随机样本。它输出一个二进制向量，其中每个令牌都是独立和均匀随机选择的。
- `perturb_interpretable_space`，扰动的样品是否在可解释的空间中。`LimeBase`还支持在原始输入空间中进行采样，但我们不需要它。
- `from_interp_rep_transform`，该函数将扰动的可解释样本转换回原始输入空间。如上所述，这个论点是我们使用`LimeBase`的主要原因。我们根据可解释的表示从原始文本输入中选择当前令牌的子集。
- `to_interp_rep_transform`，与`from_interp_rep_transform`相反。仅当`perturb_interpretable_space`设置为false时才需要它。

```python
# remove the batch dimension for the embedding-bag model
def forward_func(text, offsets):
    return eb_model(text.squeeze(0), offsets)

# encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
    original_emb = eb_model.embedding(original_inp, None)
    perturbed_emb = eb_model.embedding(perturbed_inp, None)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
    return torch.exp(-1 * (distance ** 2) / 2)

# binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, **kwargs):
    probs = torch.ones_like(text) * 0.5
    return torch.bernoulli(probs).long()

# remove absenst token based on the intepretable representation sample
def interp_to_input(interp_sample, original_input, **kwargs):
    return original_input[interp_sample.bool()].view(original_input.size(0), -1)

lasso_lime_base = LimeBase(
    forward_func, 
    interpretable_model=SkLearnLasso(alpha=0.08),
    similarity_func=exp_embedding_cosine_distance,
    perturb_func=bernoulli_perturb,
    perturb_interpretable_space=True,
    from_interp_rep_transform=interp_to_input,
    to_interp_rep_transform=None
)
```

归因调用与`Lime`类相同。只需记住将虚拟批处理维度添加到文本输入中，并将偏移量放在 `additional_forward_args`，因为它不是分类的功能，而是文本输入的元数据。

```python
attrs = lasso_lime_base.attribute(
    test_text.unsqueeze(0), # add batch dimension for Captum
    target=test_labels,
    additional_forward_args=(test_offsets,),
    n_samples=32000,
    show_progress=True
).squeeze(0)

print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
```

最后，让我们创建一个简单的可视化，以突出有影响力的单词，其中绿色代表正相关性，红色代表负。

```python
def show_text_attr(attrs):
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(tokenizer(test_line), attrs.tolist())
    ]
    
    display(HTML('<p>' + ' '.join(token_marks) + '</p>'))
    
show_text_attr(attrs)
```

上述视觉化应该像下面的图像一样，模型将“体育”与许多合理的单词联系起来，如“比赛”和“奖面”。
