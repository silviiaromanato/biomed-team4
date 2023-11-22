from data import *

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    '''
    Train model.
    '''
    model.to(device)
    train_losses = []
    train_accuracies = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_f1 = 0.0
        for i, data in enumerate(train_loader):
            if len(data) == 2:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
            elif len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)
            optimizer.zero_grad()
            if len(data) == 2:
                outputs = model(inputs)
            elif len(data) == 3:
                outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            outputs = outputs.detach().cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            labels = labels.detach().cpu().numpy()
            running_accuracy += accuracy_score(labels, outputs)
            running_f1 += f1_score(labels, outputs)
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(running_accuracy/len(train_loader))
        train_f1s.append(running_f1/len(train_loader))
        model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        running_f1 = 0.0
        for i, data in enumerate(val_loader):
            if len(data) == 2:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
            elif len(data) == 3:
                inputs1, inputs2, labels = data
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)
            if len(data) == 2:
                outputs = model(inputs)
            elif len(data) == 3:
                outputs

