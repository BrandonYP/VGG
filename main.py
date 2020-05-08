import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt
import VGG


def model_train():
    correct = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        Model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)  # 将输入和目标在每一步都送入GPU
            # 训练
            outputs = Model(inputs)
            loss = criterion(outputs, labels)
            # 将梯度置零
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 优化
            # 统计数据
            if i % 50 == 49:  # 每 batchsize  张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, loss.item()))
            _, outputs = torch.max(outputs.data, 1)
            correct += (outputs == labels).sum().item()
        train_loss.append(correct/50000)
        correct = 0
        model_test()


def model_test():
    Model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    test_accuracy.append(correct / total)


if __name__ == '__main__':
    """CIFAR10 60000*32*32RGB"""
    """:parameters"""
    batch_size = 100
    learning_rate = 0.0001
    num_epochs = 20
    train_loss = []
    test_accuracy = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = time.perf_counter()

    trainset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=True,
                                            download=False,
                                            transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=False,
                                           download=False,
                                           transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False)

    Model = VGG.vgg16().to(device)
    model_train()
    # 保存
    torch.save(Model.state_dict(), '\ImageSegmentation\VGG\parameter.pkl')
    # 加载
    # Model.load_state_dict(torch.load('\ImageSegmentation\VGG\parameter.pkl'))
    # model_test()
    end = time.perf_counter()
    print(end - start)

    plt.plot(range(num_epochs), train_loss, 'r', label='train')
    plt.plot(range(num_epochs), test_accuracy, 'b', label='test')
    plt.show()
