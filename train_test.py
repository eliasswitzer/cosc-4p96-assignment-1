import torch
import numpy as np

def l2_regularization(model):
    """
    Calculates the sum of the squared weights for L2 Weight Decay Regularization
    """
    l2 = 0.0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "bias" not in name and "bn" not in name: # we only want the weight parameters
                l2 += torch.sum(parameter ** 2)
    return l2

def train(model, device, epochs, train_loader, val_loader, optimizer, criterion, overfitting_window, lambda_l2):

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "converged_epoch": None
    }

    val_loss_history = [] # to track the previous val loss
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            break
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # zero gradients

            outputs = model(images)  # forward pass

            ce_loss = criterion(outputs, labels)
            l2 = l2_regularization(model)
            loss = ce_loss + (lambda_l2 / 2.0) * l2

            loss.backward() # backpropigate error
            optimizer.step() # update parameters

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        val_loss /= total
        val_acc = correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} | ",
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )
        
        # Record Metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        # Overfitting Detection
        val_loss_history.append(val_loss)
        if len(val_loss_history) > overfitting_window: # only start checking after enough epochs have passed
            prev_val_loss_values = val_loss_history[-overfitting_window:]

            mean_val_loss = np.mean(prev_val_loss_values)
            std__val_loss = np.std(prev_val_loss_values)

            if val_loss > mean_val_loss + std__val_loss: # overfitting criterion
                early_stop = True
                metrics["converged_epoch"] = epoch
                print("Overfitting Detected! Stopping training.")

    if early_stop:
        print(f"Epoch Converged: {metrics['converged_epoch']}")

    return metrics

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    average_test_loss = test_loss / total
    test_acc = correct / total

    print(f"Average Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return average_test_loss, test_acc