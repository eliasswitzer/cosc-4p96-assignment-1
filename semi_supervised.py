import torch
from torch.utils.data import DataLoader, TensorDataset
from train_test import train

# SEMI-SUPERVISED LEARNIN: CO-TRAINING
# Custom dataset for storing labeled/pseudolabeled data for both models (makes adding to the datasets easier)
class CoTrainingDataset(TensorDataset):
    def __init__(self, images, labels):
        super().__init__(images, labels)
    
    def add(self, new_images, new_labels):
        if new_images.ndim == 3:
            new_images = new_images.unsqueeze(0)
        if new_labels.ndim == 0:
            new_labels = new_labels.unsqueeze(0)
        
        self.tensors = (
            torch.cat([self.tensors[0], new_images.cpu()], dim=0),
            torch.cat([self.tensors[1], new_labels.cpu().long()], dim=0)
        )

def split_views(x):
    # Split images into left and right halves
    _, _, W = x.shape
    mid = W // 2
    left = x[:, :, :mid]
    right = x[:, :, mid:]
    return left, right

def split_views_batch(x):
    # Split images into left and right halves (for batches)
    _, _, _, W = x.shape
    mid = W // 2
    left = x[:, :, :, :mid]
    right = x[:, :, :, mid:]
    return left, right

def co_train(model_A, model_B, device, generator, labeled_A, labeled_B, unlabeled_loader, optimizer_A, optimizer_B, criterion, overfitting_window, lambda_l2, co_training_iterations, high_confidence_threshold, low_confidence_threshold):
    for iteration in range(co_training_iterations):
        print(f"Co-training iteration {iteration + 1}")

        model_A.eval()
        model_B.eval()

        num_added_A, num_added_B = 0, 0

        with torch.no_grad():
            for x, _ in unlabeled_loader:
                x = x.to(device)
                left, right = split_views_batch(x)

                # Both models predict on unlabeled data
                probs_A = torch.softmax(model_A(left), dim=1) # softmax is here since model only outputs raw logits
                probs_B = torch.softmax(model_B(right), dim=1)

                conf_A, pred_A = probs_A.max(dim=1)
                conf_B, pred_B = probs_B.max(dim=1)

                # If Model A is confident but Model B is not, add to B's training set (and vice versa)
                add_to_A = (conf_B > high_confidence_threshold) & (conf_A < low_confidence_threshold)
                add_to_B = (conf_A > high_confidence_threshold) & (conf_B < low_confidence_threshold)

                if add_to_A.any():
                    labeled_A.add(left[add_to_A].cpu(), pred_B[add_to_A].cpu())
                    num_added_A += add_to_A.sum().item()

                if add_to_B.any():
                    labeled_B.add(right[add_to_B].cpu(), pred_A[add_to_B].cpu())
                    num_added_B += add_to_B.sum().item()

        print(f"Added {num_added_A} data examples to Model A and {num_added_B} data examples to Model B")

        if num_added_A == 0 and num_added_B == 0:
            print("No confident samples added. Ending co-training.")
            break

        # Retrain on updated datasets
        loader_A = DataLoader(labeled_A, batch_size=64, shuffle=True, generator=generator)
        loader_B = DataLoader(labeled_B, batch_size=64, shuffle=True, generator=generator)

        train(model_A, device, 5, loader_A, loader_A, optimizer_A, criterion, overfitting_window, lambda_l2)
        train(model_B, device, 5, loader_B, loader_B, optimizer_B, criterion, overfitting_window, lambda_l2)

def predict_with_both_models(model_A, model_B, loader, device):
    model_A.eval()
    model_B.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            left, right = split_views_batch(x)

            # Average the outputs
            probs = (torch.softmax(model_A(left), dim=1) + torch.softmax(model_B(right), dim=1)) / 2

            preds = probs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total