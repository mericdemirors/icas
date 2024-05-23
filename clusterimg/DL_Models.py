import torch.nn as nn

class PowerOf2s256andAbove(nn.Module):
    def __init__(self):
        super(PowerOf2s256andAbove, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            
        self.encoder5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True))
        # ------------------------------------------------------------------------------------------------------------ #
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(inplace=True))
        
        self.unpool7 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.unpool8 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.unpool9 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4))
        
        self.unpool10 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x, indices1 = self.encoder1(x)
        x, indices2 = self.encoder2(x)
        x, indices3 = self.encoder3(x)
        x, indices4 = self.encoder4(x)
        x = self.encoder5(x)
        x = self.decoder6(x)
        x = self.unpool7(x, indices4)
        x = self.decoder7(x)
        x = self.unpool8(x, indices3)
        x = self.decoder8(x)
        x = self.unpool9(x, indices2)
        x = self.decoder9(x)
        x = self.unpool10(x, indices1)
        x = self.decoder10(x)
        return x
    
    def embed(self, x):
        """function to get image embeddings from models encoder part

        Args:
            x (torch.tensor): image batch

        Returns:
            torch.tensor: image embeddings
        """
        x, _ = self.encoder1(x)
        x, _ = self.encoder2(x)
        x, _ = self.encoder3(x)
        x, _ = self.encoder4(x)
        x = self.encoder5(x)
        x = x.view(x.size(0), -1)
        return x
    
class PowerOf2s32to128(nn.Module):
    def __init__(self):
        super(PowerOf2s32to128, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True))
        # ------------------------------------------------------------------------------------------------------------ #
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(inplace=True))
                
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4))
        
        self.unpool6 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x, indices1 = self.encoder1(x)
        x, indices2 = self.encoder2(x)
        x = self.encoder3(x)
        x = self.decoder4(x)
        x = self.unpool5(x, indices2)
        x = self.decoder5(x)
        x = self.unpool6(x, indices1)
        x = self.decoder6(x)
        return x
    
    def embed(self, x):
        """function to get image embeddings from models encoder part

        Args:
            x (torch.tensor): image batch

        Returns:
            torch.tensor: image embeddings
        """
        x, _ = self.encoder1(x)
        x, _ = self.encoder2(x)
        x = self.encoder3(x)
        x = x.view(x.size(0), -1)
        return x