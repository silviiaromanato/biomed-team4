from data import *

def train_model(model, num_epochs, train_dataloader, criterion, optimizer, device, verbose = False):
    print('Training model...')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (x_pa, x_lateral, labels) in enumerate(train_dataloader):
            # Forward pass
            inputs = x_lateral.to(device), x_pa.to(device)
            outputs = model(inputs[0], inputs[1])
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            if verbose:
                print('Predicted: ', predicted, 'Labels: ', labels, 'Loss: ', loss.item())
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100 * correct_predictions / total_predictions

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return model

def test_model(model, test_dataloader, device):
    print('Testing model...')
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, (x_pa, x_lateral, labels) in enumerate(test_dataloader):
            inputs = x_lateral.to(device), x_pa.to(device)
            labels = labels.to(device)
            outputs = model(inputs[0], inputs[1])
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        test_accuracy = 100 * correct_predictions / total_predictions
        print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    return test_accuracy, model

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

