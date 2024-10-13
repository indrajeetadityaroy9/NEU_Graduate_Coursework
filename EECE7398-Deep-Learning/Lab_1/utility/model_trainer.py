import torch
from tqdm import tqdm


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=500):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (GPU) for training")
    else:
        device = torch.device('cpu')
        print("MPS not available, using CPU for training")

    model.to(device)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)

        for batch_idx, (inputs, labels) in train_bar:
            # Move inputs and labels to the GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log training loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update the tqdm progress bar
            train_bar.set_postfix({
                'Loss': running_loss / (batch_idx + 1),
                'Accuracy': 100. * correct / total
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Testing Phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", leave=False)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                test_bar.set_postfix({
                    'Loss': test_loss / (batch_idx + 1),
                    'Accuracy': 100. * correct / total
                })

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        print(f"{epoch}/{epochs:<5} | {train_loss:<10.4f} | {train_acc:<12.4f} | {test_loss:<10.4f} | {test_acc:<10.4f}")

        scheduler.step()

        if epoch == epochs:
            model_path = ".model_state_dict.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model weights saved in file: {model_path}")
