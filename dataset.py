from PIL import Image
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data 
        self.transform = transform
        
    def __getitem__(self, index):
        path, label = self.data[index]
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.data)
