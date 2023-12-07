'''
Training and evaluation functions. 

TODO: Add validation set, scheduler and log results to W&B. 
'''

from data import *
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, accuracy_score, precision_score, recall_score


def train(model, criterion, optimizer, scheduler, train_data, val_data, num_epochs, device='cpu', verbose=False):
    print('Training model...')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (x_pa, x_lateral, labels) in enumerate(train_data):
            # Forward pass
            inputs = x_lateral.to(device), x_pa.to(device)
            outputs = model(inputs[0], inputs[1])
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            if verbose:
                print('Predicted: ', predicted, 'Labels: ', labels, 'Loss: ', loss.item())
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_data)
        epoch_accuracy = 100 * correct_predictions / total_predictions

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    train_results = {
        'train_loss': epoch_loss,
        'train_accuracy': epoch_accuracy
    }
    return train_results

def test(model, test_data, device='cpu'):
    print('Testing model...')
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, (x_pa, x_lateral, labels) in enumerate(test_data):
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


# PROBABLY DEPRECATED
def train_model(model, num_epochs, train_dataloader, criterion, optimizer, device, verbose = False):
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