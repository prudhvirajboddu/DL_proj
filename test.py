from PIL import Image

image = Image.open('train/1c620bed-IMG_8229.jpg').convert('RGB')

from torchvision import transforms


# tr = transforms.PILToTensor()

tr = transforms.Compose([
        # transforms.PILToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


print(tr(image).shape)