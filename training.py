import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from create_data_jsons import create_train_test_jsons

from neuralnet import Simple_CNN, hp, classes
from preprocess_data import TrainDataset, TestDataset, data_processing


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, _data in enumerate(train_loader):
        images, labels = _data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx + 1, running_loss / 10))
            running_loss = 0.0
            torch.save(model.state_dict(), './img_classifier.pt')


def test(model, device, test_loader):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(hp['batch_size'])))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(hp['batch_size'])))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def main():
    create_train_test_jsons()
    model = Simple_CNN()
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_dataset, test_dataset = TrainDataset(), TestDataset()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hp['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x),
                                   **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hp['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x),
                                  **kwargs)
    model.to(device)
    criterion.to(device)
    print('Total Trainable Parameters:', sum([param.nelement() for param in model.parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])

    for epoch in range(1, hp['epochs'] + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader)
    print('Training finished.')


main()
