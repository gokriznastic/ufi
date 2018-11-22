
# coding: utf-8

# In[1]:


import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from os import listdir
from os.path import isfile, join
import string


# In[2]:


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search( 
            b"(^P5\s(?:\s*#.*[\r\n])*" 
            b"(\d+)\s(?:\s*#.*[\r\n])*" 
            b"(\d+)\s(?:\s*#.*[\r\n])*" 
            b"(\d+)\s)", buffer).groups()

    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


# ## data extraction

# ### training data

# In[3]:


my_path = 'ufi-cropped\\train'
folders = [f for f in listdir(my_path)]


# In[4]:


len(folders)


# In[5]:


files = []
for folder_name in folders:
    folder_path = join(my_path, folder_name)
    files.append([f for f in listdir(folder_path) if f.endswith('.pgm')])


# In[6]:


sum(len(files[i]) for i in range(len(folders)))


# In[7]:


pathname_list = []
train_labels = []
for fo in range(len(folders)):
    for fi in files[fo]:
        pathname_list.append(join(my_path, join(folders[fo], fi)))
        train_labels.append(folders[fo])


# In[8]:


len(pathname_list)


# In[9]:


train_images = np.empty((4316, 128, 128))


# In[10]:


train_image = read_pgm(pathname_list[0])
train_image.shape


# In[11]:


for i in range(len(pathname_list)):
    train_images[i] = read_pgm(pathname_list[i])


# In[12]:


type(train_images), train_images.shape


# In[13]:


for i in range(len(train_labels)):
    train_labels[i] = train_labels[i][1:]
    train_labels[i] = int(train_labels[i])


# In[14]:


train_labels = np.asarray(train_labels)


# In[15]:


train_labels = torch.from_numpy(train_labels).long() #-- model was throwing error during training so labels had to be type casted


# In[16]:


type(train_labels), train_labels.shape


# ### testing data

# In[17]:


my_path = 'ufi-cropped\\test'
folders = [f for f in listdir(my_path)]


# In[18]:


len(folders)


# In[19]:


files = []
for folder_name in folders:
    folder_path = join(my_path, folder_name)
    files.append([f for f in listdir(folder_path) if f.endswith('.pgm')])


# In[20]:


sum(len(files[i]) for i in range(len(folders)))


# In[21]:


pathname_list = []
test_labels = []
for fo in range(len(folders)):
    for fi in files[fo]:
        pathname_list.append(join(my_path, join(folders[fo], fi)))
        test_labels.append(folders[fo])


# In[22]:


len(pathname_list)


# In[23]:


test_images = np.empty((605, 128, 128))


# In[24]:


test_image = read_pgm(pathname_list[0])
test_image.shape


# In[25]:


for i in range(len(pathname_list)):
    test_images[i] = read_pgm(pathname_list[i])


# In[26]:


type(test_images), test_images.shape


# In[27]:


for i in range(len(test_labels)):
    test_labels[i] = test_labels[i][1:]
    test_labels[i] = int(test_labels[i])


# In[28]:


test_labels = np.asarray(test_labels)


# In[29]:


test_labels = torch.from_numpy(test_labels).long() #-- model was throwing error during training so labels had to be type casted


# In[30]:


type(test_labels), test_labels.shape


# ## Input Pipeline

# In[31]:


class UFIDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return 4316
    
    def __getitem__(self, index):
        img = self.images[index]
        image = torch.from_numpy(img)
        image = image.float()
        label = self.labels[index]
        
        return image, label


# In[32]:


ufi_train_data = UFIDataset(train_images, train_labels)


# In[33]:


train_loader = torch.utils.data.DataLoader(dataset=ufi_train_data,
                                         batch_size=60,
                                         shuffle=True)


# In[34]:


ufi_test_data = UFIDataset(test_images, test_labels)


# In[35]:


test_loader = torch.utils.data.DataLoader(dataset=ufi_test_data,
                                         batch_size=60,
                                         shuffle=True)


# In[36]:


#sample batch check
data_iter = iter(train_loader)
X, Y = data_iter.next()

X.size(), Y.size()


# ## the CNN model

# In[37]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[38]:


num_classes = len(folders)
num_epochs = 10
learning_rate = 0.001


# In[39]:


# convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(32*32*64, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size()[0],-1)
        out = self.fc(out)
        
        return out


# In[40]:


model = ConvNet(num_classes).to(device)


# In[41]:


# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[42]:


# train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #labels = torch.autograd.Variable(labels.long(), requires_grad=False)
        # move tensors to device
        images = images.reshape((-1, 1, 128, 128)).to(device)
        labels = labels.to(device)
        
        #forward pass
        pred = model.forward(images)
        loss = loss_fn(pred, labels)
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if((i+1)%10 == 0):
            print('epoch[{}/{}], step[{}/{}], loss: {:4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# In[43]:


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.reshape((-1, 1, 128, 128)).to(device)
        labels = labels.to(device)
        
        pred = model.forward(images)
        
        predictions = torch.argmax(pred.data, 1)
        
        total += labels.size()[0]
        correct += (predictions == labels).sum().item()


# In[44]:


print('training-accuracy:', (correct/total)*100)


# In[45]:


# testing the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in zip(test_images, test_labels): #-- note that we aren't able to test_loader here due to some errors
        images = torch.from_numpy(images)
        images = images.float()
        images = images.reshape((-1, 1, 128, 128)).to(device)
        labels = labels.to(device)
        
        pred = model.forward(images)
        
        predictions = torch.argmax(pred.data, 1)
        
        correct += (predictions == labels).sum().item()
    total = test_labels.size()[0]


# In[46]:


print('testing-accuracy:', (correct/total)*100)


# In[47]:


# save the model checkpoint
torch.save(model, 'UFI.ckpt')


# In[48]:


# save the model params checkpoint
torch.save(model.state_dict(), 'UFI-params.ckpt')
