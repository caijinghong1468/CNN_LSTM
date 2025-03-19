import torch 
from torch import nn 
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "mps" 

#create a simple dataset using the linear regression formula of y = wx + b
weight =0.5
bias =0.5

start =  0 
end = 1 
step = 0.02


X = torch.arange(start,end,step).unsqueeze(dim=1)
Y = weight * X + bias
 
train_split = int (0.8*len(X))

X_train , Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]   


def plot_prediction(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,Predict=None):
    plt.figure(figsize=(10,6))
    plt.scatter(X_train,Y_train,c='r',s=4,label = 'Training data')
    plt.scatter(X_test,Y_test,c='g',s=4,label = 'Testing data')
    if Predict is not None:
        plt.scatter(X_test,Predict,c='b',s=4,label = 'Prediction')
    plt.legend(prop={"size":15})
    plt.show()
"""
plot_prediction(X_train,Y_train,X_test,Y_test)
   """          
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
    def forward(self,x: torch.Tensor)->torch.Tensor:
        return self.linear_layer(x)
    

torch.manual_seed(42)
model = LinearRegression()
model.to(device)
#print(next(model.parameters()).device)
 
"""traning
 -loss function
 -optimizer
 -traning loop
 -testing loop
 """


loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001)

torch.manual_seed(42)
epoches = 2000

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

for epoch in range(epoches):
    model.train()
    Y_preds = model(X_train)
    loss = loss_fn(Y_preds,Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        Y_test_preds = model(X_test)
        test_loss = loss_fn(Y_test_preds,Y_test)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss: {loss}, Test Loss: {test_loss}")

print(model.state_dict())



model.eval()
with torch.inference_mode():
    Y_test_preds = model(X_test)
#plot_prediction(Predict=Y_test_preds.cpu())


torch.save(model.state_dict(),"./model.pth")
loaded_model = LinearRegression()
loaded_model.load_state_dict(torch.load("./model.pth"))

loaded_model.to(device)

loaded_model.eval()
with torch.inference_mode():
    loaded_pred = loaded_model(X_test)
print(loaded_pred == Y_test_preds)