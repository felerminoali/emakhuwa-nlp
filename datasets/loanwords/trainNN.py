from dataload import *

import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

state = random.randint(0, 42)
print(state)

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
    
    def calculate_accuracy(self, y_true, y_pred):
        predicted = y_pred.ge(.5) 
        return ((y_true == predicted).sum().float() / len(y_true), (y_true == predicted).sum())
    
    def round_tensor(self, t, decimal_places=3):
        return round(t.item(), decimal_places)
    
    def plot_losses(self, train_losses, val_losses, train_accur, val_accur):
        epochs = range(1, len(train_accur) + 1)

        plt.plot(epochs, train_accur, 'bo', label='Training acc')
        plt.plot(epochs, val_accur, 'b', label='Vaidation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, train_losses, 'bo', label='Training loss')
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

display_fields = display + features

f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []

y_test = []
y_pred_ = []

for i in range(5):

    train_set = train_alldata[display_fields + labels]
    train_set = train_set.sample(frac=1, random_state=state)
    x_train = train_set[features].values
    x_means = np.mean(x_train, axis=0)
    x_stds = np.std(x_train, axis=0)
    x_stds[x_stds == 0] = 1
    y_train = train_set[labels].values.ravel()

    test_set = test_alldata[display_fields + labels]
    test_set = test_set.sample(frac=1, random_state=state)
    x_test = test_set[features].values
    y_test = test_set[labels].values.ravel()

    x_train = (x_train - x_means)/x_stds

    print(x_means)
    print(x_stds)

    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
        
    model = NeuralNetwork(x_train.shape[1]).to(device)
    print(model)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    #optimizer = optim.SGD(model.parameters(),lr=0.00001, momentum=0.0,  weight_decay=0.0, nesterov=False)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=state, stratify=y_train)

    x_train = torch.tensor(x_train).to(device)
    x_val = torch.tensor(x_val).to(device)
    y_train = torch.tensor(y_train).reshape(-1,1).to(device)
    y_val = torch.tensor(y_val).reshape(-1,1).to(device)
            
    print("Training on all langs\n")
    train_losses, val_losses, train_accur, val_accur = model.fit(x_train, y_train, x_val, y_val, criterion, optimizer, n_epochs=10000)
    model.plot_losses(train_losses,val_losses,train_accur,val_accur)
        
    model.eval()

    print("Evaluating on all langs")
    x_test = (x_test - x_means)/x_stds
    x_test = torch.tensor(x_test).to(device)

    with torch.no_grad():
        y_pred = model(torch.tensor(x_test).float())[0] > .5
        y_pred = y_pred.detach().cpu().numpy()
        y_pred_ = [1 if y else 0 for y in y_pred]

        f1_scores.append(f1_score(y_test, y_pred_ ))
        precision_scores.append(precision_score(y_test, y_pred_))
        recall_scores.append(recall_score(y_test, y_pred_ ))
        accuracy_scores.append(accuracy_score(y_test, y_pred_))

repot_txt = f"""
f1-score :  { np.mean(f1_scores)} { np.std(f1_scores)}
precision :  {np.mean(precision_scores)} {np.std(precision_scores)}
recall :  {np.mean(recall_scores)} {np.std(recall_scores)}
accuracy :  {np.mean(accuracy_scores)} {np.std(accuracy_scores)}
"""
print(repot_txt)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
class_report = classification_report(y_test, y_pred)

save_predition(test_set, y_test, y_pred_, split+'-NN', class_report, repot_txt)