from preprocessing import process_input, process_word_translation_match

import torch
from torch import nn
from torch import optim


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            
        )

    def forward(self, x):
        logits_new = self.linear_relu_stack(x)
        logits  = logits_new
        
        return torch.sigmoid(logits), logits_new
    
    def save_checkpoint(self, state, filename):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def fit(self, X_train, Y_train, X_val, Y_val, criterion, optimizer, n_epochs=5000):
        train_losses = []
        val_losses = []
        train_accur = []
        val_accur = []

        for epoch in range(n_epochs):
            y_pred, logits = self(X_train.float())

            train_loss = criterion(y_pred, Y_train.float())

            if epoch % (n_epochs // 50) == 0:
                train_acc,_ = self.calculate_accuracy(Y_train, y_pred)

                y_val_pred = self(X_val.float())[0]

                val_loss = criterion(y_val_pred, Y_val.float())

                val_acc, total_corr = self.calculate_accuracy(Y_val, y_val_pred)

                print(f'''epoch {epoch}
                    Train set - loss: {self.round_tensor(train_loss)}, accuracy: {self.round_tensor(train_acc)} 
                    Val set - loss: {self.round_tensor(val_loss)}, accuracy: {self.round_tensor(val_acc)}''')
                
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                model_path = '../../checkpoints/nn.pth.tar'
                self.save_checkpoint(checkpoint, model_path)

                train_losses.append(train_loss.detach().cpu().numpy())
                val_losses.append(val_loss.detach().cpu().numpy())

                val_accur.append(val_acc.detach().cpu().numpy())
                train_accur.append(train_acc.detach().cpu().numpy())

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()
            
        return train_losses,val_losses,train_accur,val_accur


def predict(word):
    word, translation, word_match = process_word_translation_match(word)
    inputs = process_input(word)

    model = NeuralNetwork(inputs.shape[1]).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    model_path = 'model/nn.pth.tar'
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    model_ = model(torch.tensor(inputs).float())

    if model_[0] > .9:
        return (word, translation, model_[0].item())
    
    return None