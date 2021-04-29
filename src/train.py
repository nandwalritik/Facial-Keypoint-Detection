import torch 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import matplotlib 
import config 
import utils 

from model import FaceKeypointModel 
from dataset import train_data,train_loader,valid_data,valid_loader
from tqdm import tqdm 
matplotlib.style.use('ggplot')

model = FaceKeypointModel().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(),lr=config.LR)
criterion = nn.MSELoss()

def fit(model,dataloader,data):
    print("\n--------Training-----------\n")
    model.train()
    train_running_loss=0.0
    counter = 0
    # calculating number of batches 
    num_batches = int(len(data)/dataloader.batch_size)
    for i,data in tqdm(enumerate(dataloader),total=num_batches):
        counter += 1
        image,keypoints = data['image'].to(config.DEVICE),data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0),-1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs,keypoints)
        train_running_loss += loss.item()
        loss.backward() 
        optimizer.step()
    train_loss = train_running_loss/counter 
    return train_loss

def validate(model,dataloader,data,epoch):
    print('\n------Validating------\n')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculating number of batches 
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i,data in tqdm(enumerate(dataloader),total=num_batches):
            counter += 1
            image,keypoints = data['image'].to(config.DEVICE),data['keypoints'].to(config.DEVICE)
            keypoints = keypoints.view(keypoints.size(0),-1)
            outputs = model(image)
            loss = criterion(outputs,keypoints)
            valid_running_loss += loss.item()
            if (epoch+1) % 25 == 0 and i == 0:
                utils.valid_keypoints_plot(image,outputs,keypoints,epoch)
    
    valid_loss = valid_running_loss/counter
    return valid_loss 


train_loss = []
val_loss = []
for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(model,train_loader,train_data)
    val_epoch_loss = validate(model,valid_loader,valid_data,epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')

# loss plots
plt.figure(figsize=(10,7))
plt.plot(train_loss,color="orange",label='train loss')
plt.plot(val_loss,color="red",label='validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{config.output_path}/loss.png")
plt.show()
torch.save({
    'epoch':config.EPOCHS,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss':criterion,
    },f"{config.output_path}/model.pth")
    
print("\n---------DONE TRAINING----------\n")