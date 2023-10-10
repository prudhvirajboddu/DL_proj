import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(label_path).convert('RGB')
        
        ymin = self.annotations.iloc[index, 1] 
        xmin = self.annotations.iloc[index, 2]
        ymax = self.annotations.iloc[index, 3]
        xmax = self.annotations.iloc[index, 4]
        boxes = [[xmin, ymin, xmax, ymax]]
        
        if self.transform:
            image = self.transform(image)
            
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor([1]) 

        return image, target
    

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(transforms.PILToTensor()),
        transforms.RandomHorizontalFlip(),
        transforms.Scale((224,224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomRotation(5, resample=False,expand=False, center=None),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Scale((224,224)),
       transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomRotation(5, resample=False,expand=False, center=None),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}
    


dataset = FaceDataset('train_labels.csv','train',transform = transforms.Compose([transforms.PILToTensor()]))

train_loader = DataLoader(dataset,shuffle=True)

for images,targets in train_loader:
    print(images.shape)
    print(targets)
    break
