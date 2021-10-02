# Visual Place Recognition with NetVLAD

[![stars](https://img.shields.io/github/stars/socome/NetVLAD-in-Colab?style=social)](https://github.com/socome/NetVLAD-in-Colab/stargazers)

#### [What is the Visual Place Recognition(in Korean) - 3rd Place Solution to NAVER LABS Mapping & Localization](https://youtu.be/pkbkJ4R8Hh8)

### How to run

#### Step 1. Downloading a source code(notebook) and Berlin Kudamm dataset

[![Base](https://img.shields.io/badge/Download-Basecode%20%26%20Datasets-yellow)](https://drive.google.com/file/d/1b8UKHViSrZ2mbT27DqQxDj0D_xnLq1pC/view)

#### Step 2. Execute all code cells in the notebook (click on the notebook toolbar or press Ctrl + Shift + Alt + Enter)

##### Step 2-1. Mount the Google Drive to Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

##### Step 2-2. Install the [faiss-gpu](https://github.com/facebookresearch/faiss) for Nearest Neighbor search

```
!pip install faiss-gpu
```

##### Step 2-3. Define the NetVLAD (This code is built mostyl based on [NetVLAD-pytorch](https://github.com/Nanne/pytorch-NetVlad)

```
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=12, dim=128, alpha=100.0,normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )

    def forward(self, x):

        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
```

##### Step 2-3. Define VPR Model (CNN(VGG16) + NetVLAD)

![image](https://user-images.githubusercontent.com/44772344/135706112-2c038384-b77f-43cd-b568-e47b70e355b8.png)


```
# VGG 16
encoder = vgg16(pretrained=True)
layers = list(encoder.features.children())[:-2]

for l in layers[:-5]: 
    for p in l.parameters():
        p.requires_grad = False

model = nn.Module() 

encoder = nn.Sequential(*layers)
model.add_module('encoder', encoder)

dim = list(encoder.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=16, dim=dim)
model.add_module('pool', net_vlad)

model = model.cuda()
```

##### Step 2-4. Load the pre-trained weights from checkpoint, which was provided from [here](https://github.com/Nik-V9/HEAPUtil)

```
load_model = torch.load('./pittsburgh_checkpoint.pth.tar')
model.load_state_dict(load_model['state_dict'])
```


##### Step 2-5. Creating a Dataloader (This code is built mostyl based on [here](https://github.com/Nik-V9/HEAPUtil))

```

def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat['dbStruct'][0]

    dataset = 'dataset'

    whichSet = 'VPR'

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage, 
            locQ, numDb, numQ, posDistThr, 
            posDistSqThr)
  
dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
  'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
  'posDistThr', 'posDistSqThr'])
  
class BerlinDataset(data.Dataset) :
  
  def __init__(self,condition='train') :
    self.dbStruct = parse_dbStruct('/content/drive/My Drive/컴퓨터비전/NetVLAD/berlin.mat') # 필요시 경로수정
    self.input_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
      ])
    
    self.condition = condition

    if self.condition == 'train' :
      self.images = [join(root_dir, dbIm.replace(' ','')) for dbIm in self.dbStruct.dbImage]
    elif self.condition == 'test' :
      self.images = [join(root_dir, qIm.replace(' ','')) for qIm in self.dbStruct.qImage]
    else :
      self.images = [join(root_dir, dbIm.replace(' ','')) for dbIm in self.dbStruct.dbImage]
    
    self.positives = None
    self.distances = None

    self.getPositives()
  
  def __getitem__(self, index):

      if self.condition == 'train' :
        img = Image.open(self.images[index])
        img = self.input_transform(img)

        pos_list = self.positives[index].tolist()
        pos_list.remove(index)
        pos_index = self.positives[index][np.random.randint(0,len(self.positives[index]))]
        pos_img = Image.open(self.images[pos_index])
        pos_img = self.input_transform(pos_img)

        pos_list = pos_list + [index]
        neg_index = choice([i for i in range(len(self.images)) if i not in pos_list])
        neg_img = Image.open(self.images[neg_index])
        neg_img = self.input_transform(neg_img)

        img = torch.stack([img,pos_img,neg_img],dim=0)
        label = torch.Tensor([0, 0, 1])

        return img, label

      elif self.condition == 'test' :
        img = Image.open(self.images[index])
        img = self.input_transform(img)

        return img
      
      else :
        img = Image.open(self.images[index])
        img = self.input_transform(img)

        return img


  def __len__(self):
      return len(self.images)

  def getPositives(self):
      # positives for evaluation are those within trivial threshold range
      #fit NN to find them, search by radius
      if  self.condition == 'train' :
          knn = NearestNeighbors(n_jobs=1)
          knn.fit(self.dbStruct.locDb)

          self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locDb,radius=self.dbStruct.posDistThr)
      else :
          knn = NearestNeighbors(n_jobs=1)
          knn.fit(self.dbStruct.locDb)

          self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locQ,
                  radius=self.dbStruct.posDistThr)
      
      return self.positives
```

##### Step 2-6. Fine-tuning with Berlin Kudamm dataset

```
epochs = 5
global_batch_size = 8
lr = 0.00001
momentum = 0.9
weightDecay = 0.001
losses = AverageMeter()
best_loss = 100.0
margin = 0.1 

criterion = nn.TripletMarginLoss(margin=margin**0.5, p=2, reduction='sum').cuda()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weightDecay)

model.train()

for epoch in range(epochs):
  for batch_idx, (train_image,train_label) in enumerate(train_loader) :
    output_train = model.encoder(train_image.squeeze().cuda())
    output_train = model.pool(output_train)
    triplet_loss = criterion(output_train[0].reshape(1,-1),output_train[1].reshape(1,-1),output_train[2].reshape(1,-1))

    if batch_idx == 0 :
      optimizer.zero_grad()

    triplet_loss.backward(retain_graph=True)
    losses.update(triplet_loss.item())

    if (batch_idx +1) % global_batch_size == 0 :
      for name, p in model.named_parameters():
        if p.requires_grad:
          p.grad /= global_batch_size
    
        optimizer.step()
        optimizer.zero_grad()

    if batch_idx % 20 == 0 and batch_idx != 0:
      print('epoch : {}, batch_idx  : {}, triplet_loss : {}'.format(epoch,batch_idx,losses.avg))
  
  if best_loss > losses.avg :
    best_save_name = 'best_model.pt'
    best_path = F"./ckpt/{best_save_name}" 
    torch.save(model.state_dict(), best_path)
    
  model_save_name = 'model_{:02d}.pt'.format(epoch)
  path = F"./ckpt/{model_save_name}" 
  torch.save(model.state_dict(), path)
```
##### Step 2-7. Extract NetVLAD descriptors from Reference and Query

```
from tqdm import tqdm

cluster_dataset = BerlinDataset(condition="cluster")
cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=1,shuffle=False,num_workers=0) 

train_feature_list = list()

model.eval()

with torch.no_grad():
  for batch_idx, train_image in tqdm(enumerate(cluster_loader)) :
    output_train = model.encoder(train_image.cuda())
    output_train = model.pool(output_train)
    train_feature_list.append(output_train.squeeze().detach().cpu().numpy())

train_feature_list = np.array(train_feature_list)
```

```
test_dataset = BerlinDataset(condition="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=0) 

test_feature_list = list()

with torch.no_grad():
  for batch_idx, test_image in tqdm(enumerate(test_loader)) :
    output_test = model.encoder(test_image.cuda())
    output_test = model.pool(output_test)
    test_feature_list.append(output_test.squeeze().detach().cpu().numpy())

test_feature_list = np.array(test_feature_list)

```

##### Step 2-8. Predict the top-20 highest probability reference indices using fassi

```
import faiss

n_values = [1,5,10,20]

faiss_index = faiss.IndexFlatL2(train_feature_list.shape[1])
faiss_index.add(train_feature_list)
_, predictions = faiss_index.search(test_feature_list, max(n_values))


```


#### Step 3. Submit to [EvalAI for Class](http://203.250.148.129:3088/web/challenges/challenge-page/45/submission)

##### Step 3-1. Make 'submission.json'

```
import json

file_path = "./submit.json"

data = {}
data['Query'] = list()

for i in range(len(predictions)) :
  data_t = [("id",i),("positive",predictions[i].tolist())]
  data_t = dict(data_t)
  data['Query'].append(data_t)
  
with open(file_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)
```

##### Step 3-2. Submit to EvalAI

![image](https://user-images.githubusercontent.com/44772344/135705861-fa0157b2-35cc-4261-a28d-5fa452c4163a.png)


### Reference code

[NetVLAD-pytorch](https://github.com/Nanne/pytorch-NetVlad) <br/>
[A Hierarchical Dual Model of Environment- and Place-Specific Utility for Visual Place Recognition](https://github.com/Nik-V9/HEAPUtil)

