# NN Lab

注意：

1. Autoencoder.py, random_generation.py, VarAutoencoder.py, train_script.py中的代码均为选做部分（Part 3和Part 4），在Part 1与Part 2部分中，你无需阅读或修改。
2. 为了正常运行data_processor等，请你安装torch与torchvision依赖，安装说明见[PyTorch官方页面](https://pytorch.org/get-started/locally/)。


## Part 1: 手写反向传播算法 (12.5 pts)

在这一部分中，你将手写多种不同层的反向传播算法，具体包括Linear Layer、ReLU、Sigmoid以及MSE损失层。请采用Numpy来完成这一部分的代码。

我们提供了一个自动微分（Autograd）框架，详见`./autograd`目录。在`./autograd/BaseGraph.py`中，你会看到一个用于实现一维直线型计算图 (high-level) 的计算图类。同时，在`./autograd/BaseNode.py`中，我们提供了计算图节点的基类。

接下来，请你：

1. 阅读`./autograd/BaseGraph.py`和`./autograd/BaseNode.py`，理解BaseGraph和BaseNode的类方法等，进而完成BaseGraph中的backward和optimstep函数。
2. 实现`./autograd/Nodes.py`中各类计算图节点（如上文提到的Linear Layer等）的前向传播（forward）和反向传播（backward）函数。


## Part 2: MLP Autoencoder (12.5 pts)

本部分提供了一个简易的数据集，存放在`./flowers`文件夹下。该数据集包含3种不同的花，每种有～1000张图片。你将使用第1部分中实现的计算节点构建一个MLP Autoencoder， 在此数据集上进行训练，以得到一个可以对flower图片进行压缩编码与重建的autoencoder。

请你：

1. 补全`./MLPAE_train_script.py`中的模型代码，包括Autoencoder的encoder和decoder部分。

   **要求：**数据维度变化如下：3\*H\*W -> 256 -> 128 -> 64 (Latent vector) -> 128 -> 256 -> 3\*H\*W；使用ReLU作为激活函数；确保输出值位于[0, 1]范围内。

2. 补全`./MLPAE_train_script.py`中的Training Loop, Validation Loop以及model saving和early Stopping相关的代码。

在完成上述部分后，你可以运行`python MLPAE_train_script.py`来对模型进行训练。如果你的代码实现无误，该脚本将正常运行，并且你将观察到输出的loss逐步减少。根据你实现的代码，每当validation loss达到一个更低值时，相应的最优模型将被保存至`./models/BestModel_MLPAE.pkl`。

请你：

3. **确保最终的train loss和validation loss均小于0.03**（若你的代码无误，无需调整任何其他参数即可达到该要求）。

接下来，你可以通过运行`python visualization.py --model MLPAE`来可视化模型的输出效果。该脚本将生成的图片保存在`./vis/train_MLPAE.png`和`./vis/valid_MLPAE.png`中。你将看到原始图片与模型重建的图片之间的对比，以此来可视化模型的重建效果。



## 提交格式

提交时：
1. 请保留**除`./flowers`**以外的文件/文件夹。
2. 请确保`./model`文件夹中包含你训练得到的`BestModel_MLPAE.pkl`；`./vis`文件夹中包含`train_MLPAE.png`和`valid_MLPAE.png`；（若你同时做了附加题，请同样确保相应文件夹中包含模型`BestModel_AE(VAE).pth`和图片train(valid)\_AE(VAE).png与random_images_AE(VAE).png）
3. 请将最后的**模型训练过程输出**通过重定向保存在`./log/train_MLPAE.txt`中。具体方法为：`python MLPAE_train_script.py > ./log/train_MLPAE.txt`；（若你同时做了附加题，请将其输出保存在`./log/train_AE.txt`和`./log/train_VAE.txt`中）





---
**以下部分均为选做部分，若你对Pytorch的使用较为熟悉，同时具有一定的计算资源，可以来尝试完成。**


## Part 3: Convolutional Autoencoder (7.5 bonus pts)

在第二部分中，你可能已经观察到，MLP Autoencoder在flower数据集上的图像重建效果并不理想。这主要是因为MLP Autoencoder没有考虑到图片的空间结构，而将其当作一串向量来进行编码。回忆你在课程中学到的知识，卷积神经网络（CNN）能够更好地提取图片的深层特征，因为它具有平移不变性（translation invariance）和局部性（locality）。在本部分，你将学习使用Pytorch搭建一个Convolutional Autoencoder模型，并在flower数据集上进行训练和评估。

请你：

1. 完成`Autoencoder.py`中Convolutional Autoencoder的模型架构。

   **要求：**数据维度变化如下：3 * H * W -> 32 * H/2 * W/2 -> 64 * H/4 * W/4 -> 128 * H/4 * W/4 -> encoding_dim -> 128 * H/4 * W/4 -> 64 * H/4 * W/4 -> 32 * H/2 * W/2 -> 3 * H * W；使用Linear layer实现图片到latent vector的转换；采用Max Pooling；采用ReLU作为激活函数；确保输出值位于[0, 1]范围内。

   **你可能用到的函数有：**`torch.nn.Conv2d`、`torch.nn.MaxPool2d`、`torch.nn.Linear`、`torch.nn.ConvTranspose2d`等，请通过[pytorch官方文档][https://pytorch.org/docs/stable/index.html]来查阅、学习相关函数的用法。

在完成上述部分后，你可以运行`python train_script.py --model AE`来进行模型训练。在训练过程中，你将观察到模型损失的逐步下降，并且在其他条件相同下（均采用SGD、相同的encoding size等），Convolutional Autoencoder的train loss与validation loss均比MLP Autoencoder要低（若你的代码实现正确，在60个epoch后train loss大致会收敛到0.015以下），验证了该架构的优越性。

然而，你会发现其收敛有些慢——事实上，SGD (without momentum)并不是目前最先进的优化算法。其他算法如SGD with momentum、Adam等能够使得模型更快地收敛，并且通常能够收敛到更优点。

请你：

2. 了解SGD with momentum、Adam等优化器的原理以及torch中的调用方式，调整`train_script.py`中optimizer的类型或参数，以及scheduler的参数，以**确保模型可以在15个epoch以内，train loss降低到0.01以下，validation loss降低到0.012以下。**（提示：在引入momentum或更高阶的优化算法后，你可能需要将learning rate调低若干个数量级。）

每一次训练，最优模型都将会被保存在`./models/BestModel_AE.pth`中。你可以运行`python visualization.py --model AE`来可视化模型的重建效果，输出图片保存在`./vis/train_AE.png`与`./vis/valid_AE.png`中。你将看到原始图片与模型重建的图片的对比，以此来判断模型的重建效果。


接下来，请你回忆Autoencoder的作用：encoder部分将一个图片压缩编码成为一个latent vector，该vector可以看作是图片的深层特征；decoder部分将这个vector重建回图片。因此，我们可以用Autoencoder的decoder部分来进行图片的生成：通过随机采样一个latent vector，我们可以期待decoder部分将这个vector转换成一个有意义的图片。

请你：

3. 补全`./random_generation.py`中的关于随机采样生成图片的代码。

请运行`python random_generation.py --model AE`来生成一些图片，该脚本将会在$N(0, 1)$中随机采样若干vector，并feed进你刚刚训练好的decoder将其转换成图片，输出图片保存在`./vis/random_images_AE.png`中。

## Part 4: Variational Autoencoder (7.5 bonus pts)
在第三部分的最后，你尝试通过在标准正态分布中采样向量，并将其输入到训练好的解码器中，从而生成了一些图片。但是，你可能会发现这些图片并不是特别“有意义”：他们看起来只是不同颜色像素的随机组合，而不是我们从未见过的“花”。这是因为传统的AE并不会将数据点映射到一个有意义的分布上，而是映射到一个个离散的点上。因此，当你随机采样时，采样到“有意义”的点的概率非常低。在本部分，你将搭建一个Variational Autoencoder，以解决这一问题。

请你：

1. 补全`VarAutoencoder.py`中VarAutoencoder的模型框架。

2. 补全`VarAutoencoder.py`中VarAutoencoder的VAE_loss_function。

在完成上述部分后，你可以运行`python train_script.py --model VAE`来进行模型训练。最优模型（validation loss最低）将被保存在`./models/BestModel_VAE.pth`中。

3. 请你**确保模型在20个epoch以内，train loss降低到800.以下，validation loss降低到800.以下**。



接下来，你可以运行`python visualization.py --model VAE`来可视化模型的重建效果，输出图片保存在`./vis/train_VAE.png`与`./vis/valid_VAE.png`中。同时，你可以运行`python random_generation.py --model VAE`来生成随机一些图片，该脚本将会在$N(0, 1)$中随机采样若干vector，并通过你训练好的decoder将其转换成图片，输出图片保存在`./vis/random_images_VAE.png`中。

这次，你看到的生成图片应该会比Convolutional Autoencoder生成的图片更加“有意义”：这些图片看起来更像是真实的花，而不是像素的随机组合，虽然它们仍旧有些模糊。这其中的原因有很多，例如：1. 考虑到同学们的机器性能，我们仅采用了训练集中的一小部分，因此数据点较为不足；同时，我们将训练图片resize到了24 * 24，因此训练图片丢失了很多细节；另外，我们将模型的encoding size限制在了64，这一维度可能不足以编码图片的所有信息。2. 另一方面，VAE在生成图片上本身便具有模型设计上的一些缺陷。

如果你具有一定的计算资源，同时也对生成更高质量的图片感兴趣：

4. (不计分) 请你通过调整encoding size、训练图片的size（见create_flower_dataloaders的输入参数）、训练集的大小（full dataset可以从[这里][https://www.kaggle.com/datasets/l3llff/flowers]获取），乃至模型架构（考虑GAN、Diffusion Models等）等各种参数，以生成更高质量的图片。



