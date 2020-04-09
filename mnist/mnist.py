import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2))
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x

def main():
    
    # for GPU GTX-1050
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5,],std=[0.5,])])
    
    data_train = datasets.MNIST(root = "./data/", transform = transform, train = True, download = True)
    data_test = datasets.MNIST(root="./data/", transform = transform, train = False)
    
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size = 64, shuffle = True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size = 64, shuffle = True)
    
    model = Model().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    #print(model)
    
    epoch_n = 5
    for epoch in range(epoch_n):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, epoch_n))
        
        for data in data_loader_train:
            X_train, y_train = data
            X_train, y_train = Variable(X_train.to(device)), Variable(y_train.to(device)) 
            outputs = model(X_train)
            _,pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0
        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test.to(device)), Variable(y_test.to(device))
            outputs = model(X_test)
            _,pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
        print("Loss:{:.6f}, Train Accuracy:{:.6f}%, Test Accuracy:{:.6f}".format(running_loss/len(data_train),
                                                                                 100*running_correct/len(data_train),
                                                                                 100*testing_correct/len(data_test)))
        print("-"*15)

if __name__ == '__main__':
    main()