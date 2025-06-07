import finding_mean_and_std
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.ops import Conv2dNormActivation
from matplotlib import pyplot as plt

torch.manual_seed(42)
torch.cuda.manual_seed(42)
preprocess = transforms.Compose(
    [
        transforms.Resize((224,224), antialias=True),
        transforms.ToTensor()
    ]
)
common_transforms = transforms.Compose(
    [
        preprocess,
        # transforms.Normalize(mean=finding_mean_and_std.means,std=finding_mean_and_std.stds)
    ]
)
train_transforms = transforms.Compose(
    [
        preprocess,
        transforms.RandomHorizontalFlip(),

        transforms.RandomErasing(p=0.3),
        # transforms.RandomApply([
        # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        #  ], p =0.1),

        # transforms.Normalize(mean = finding_mean_and_std.means,std = finding_mean_and_std.stds)
    ]
)


root_dir = Path(__file__).parent

train_data = datasets.ImageFolder(
    root=root_dir / "10_Monkey_Species/training/training",
    transform=train_transforms
)
test_data = datasets.ImageFolder(
    root=root_dir/"10_Monkey_Species/validation/validation",
    transform=common_transforms
)
train_dataloader = DataLoader(dataset=train_data,shuffle=True,batch_size=32)
test_dataloader = DataLoader(dataset=test_data,shuffle=False,batch_size=32)
# print(len(train_data),len(test_data))
# print(next(iter(train_dataloader)))
# plt.imshow(train_data[5][0].permute(1,2,0))
# plt.show()
class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self._model = torch.nn.Sequential(

        torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(inplace = True),

        torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(inplace = True),
        torch.nn.MaxPool2d(kernel_size = 2),

        torch.nn.LazyConv2d(out_channels = 64, kernel_size = 3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace = True),

        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace = True),
        torch.nn.MaxPool2d(kernel_size = 2),

        Conv2dNormActivation(in_channels = 128, out_channels=256, kernel_size = 3),

        Conv2dNormActivation(in_channels = 256, out_channels=256, kernel_size = 3),
        torch.nn.MaxPool2d(kernel_size = 2),

        Conv2dNormActivation(in_channels = 256, out_channels=512, kernel_size = 3),
        torch.nn.MaxPool2d(kernel_size = 2),

        #---------------------- Feed Forward Layers --------------------
        torch.nn.AdaptiveAvgPool2d(output_size=(3,3)),

        #------------------------------------
        # Flatten the convolutional features.
        #------------------------------------
        torch.nn.Flatten(),

        #--------------------
        # Classification Head.
        #--------------------
        torch.nn.Linear(in_features = 512*3*3, out_features = 256),
        torch.nn.Linear(in_features = 256, out_features = 10)
    )

  def forward(self,x):
      return self._model(x)
if __name__ == "__main__":
  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MyModel()
    if (root_dir / "model_weights.pth").exists:
        model.load_state_dict(torch.load(root_dir / "model_weights.pth"))
    model.to(device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-4)


    for epoch in range(30):
        model.train()
        accuracy = torch.tensor(0,device=device)
        train_loss = 0
        train_accuracy = 0
        for batch, (image,label) in enumerate(train_dataloader):
            image,label = image.to(device="cuda:0"),label.to(device="cuda:0")
            y_pred = model(image)
            y_train_pred = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
            accuracy+=(y_train_pred==label).sum()
            loss = loss_fn(y_pred,label)
            train_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss /= len(train_dataloader)
            accuracy = accuracy.item() / len(train_data)
        print(f"{epoch+1} Epoch : Train Loss is {train_loss}. Accuracy is {accuracy}")
    with torch.inference_mode():
        model.eval()
        accuracy = torch.tensor(0,device=device)
        for  batch, (image,label) in enumerate(test_dataloader):
            image = image.to(device=device)
            label = label.to(device=device)
            y_testpred = model(image)
            y_testpred = torch.argmax(torch.softmax(y_testpred,dim=1),dim=1)
            accuracy+= (y_testpred==label).sum()
            accuracy= accuracy.item() / len(test_data)
            print(f"Test Accuracy :{accuracy}")    
        
    torch.save(model.state_dict(), root_dir / "model_weights.pth")