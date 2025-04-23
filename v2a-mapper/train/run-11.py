# %%
import torch
import os
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# %%
root = "/mnt/new_volume2/vgg_sound_emb"
partition = "train"
data_dir = f"{root}/{partition}"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
root = "/mnt/new_volume2/vgg_sound_emb"
partition = "train"
data_dir = f"{root}/{partition}"

class LargeVideoDataset(Dataset):
    def __init__(self, data_dir, subset_ratio = 0.2, transform=None):
        """
        root_dir: 保存所有 .pth 文件的目录，每个文件对应一个 sample。
        transform: 如果需要对数据做预处理，可在这里传入。
        """
        super().__init__()
        # 仅收集当前目录下所有的 pth 文件列表
        file_list = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".pth"):
                    file_list.append(os.path.join(root, file))

        # 仅使用前 20% 的数据
        num_samples = int(len(file_list) * subset_ratio)

        self.file_paths = sorted(file_list)[:num_samples]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 在这里按需读取，而不是一次性加载全部
        pth_path = self.file_paths[idx]
        sample_data = torch.load(pth_path)  
        clip_feat = sample_data['clip_features']  # (64, 512)
        clap_feat = sample_data['clap_features']  # (1, 512)

        if self.transform:
            clip_feat, clap_feat = self.transform((clip_feat, clap_feat))

        return clip_feat, clap_feat

from functools import lru_cache

class LRUVideoDataset(Dataset):
    def __init__(self, data_dir, subset_ratio=0.2, cache_size=2000, transform=None):
        super().__init__()
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            for fn in files:
                if fn.endswith(".pth"):
                    file_list.append(os.path.join(root, fn))
        file_list = sorted(file_list)
        num_samples = int(len(file_list) * subset_ratio)
        self.file_paths = file_list[:num_samples]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    @lru_cache(maxsize=2000)
    def _load(self, idx):
        data = torch.load(self.file_paths[idx], map_location="cpu")
        return data["clip_features"], data["clap_features"]

    def __getitem__(self, idx):
        clip_feat, clap_feat = self._load(idx)
        if self.transform:
            clip_feat, clap_feat = self.transform((clip_feat, clap_feat))
        return clip_feat, clap_feat




# %%
class InMemoryVideoDataset(Dataset):
    def __init__(self, data_dir, subset_ratio=0.2, transform=None):
        """
        一次性把前 subset_ratio 比例的数据全部加载到内存里。
        """
        super().__init__()
        # 收集所有 .pth 文件
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            for fn in files:
                if fn.endswith(".pth"):
                    file_list.append(os.path.join(root, fn))
        file_list = sorted(file_list)
        num_samples = int(len(file_list) * subset_ratio)
        self.file_paths = file_list[:num_samples]
        self.transform = transform

        # 一次性加载到内存
        self.data = []
        print(f"Loading {len(self.file_paths)} samples into memory...")
        for path in tqdm(self.file_paths, desc="Caching data"):
            sample = torch.load(path, map_location="cpu")
            # 只保留 tensor，丢掉其它 metadata
            clip_feat = sample["clip_features"]  # (64,512)
            clap_feat = sample["clap_features"]  # (1,512)
            self.data.append((clip_feat, clap_feat))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_feat, clap_feat = self.data[idx]
        if self.transform:
            clip_feat, clap_feat = self.transform((clip_feat, clap_feat))
        return clip_feat, clap_feat


# %%


# %%
# vgg_sound = LargeVideoDataset(data_dir, subset_ratio = 0.2)
vgg_sound = InMemoryVideoDataset(data_dir, subset_ratio=0.05)

# %% [markdown]
# # DataLoader

# %%
val_ratio = 0.1
test_ratio = 0.1

total_len = len(vgg_sound)
val_len = int(total_len * val_ratio)
test_len = int(total_len * test_ratio)
train_len = total_len - val_len - test_len
train_dataset, val_dataset, test_dataset = random_split(
    vgg_sound, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42), 
)


# %%
batch_size = 128
num_workers = 8

# %%
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True
)

# %%
clip_feat, clap_feat = next(iter(train_loader))
print("Clip:", clip_feat.device, clip_feat.dtype)
print("Clap:", clap_feat.device, clap_feat.dtype)

# %% [markdown]
# # Wandb

# %%
import wandb

# %% [markdown]
# # Model

# %%
class V2AMapper(nn.Module):
    def __init__(self, input_dim, output_dim, expansion_rate=4):
        super(V2AMapper, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * expansion_rate)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(input_dim * expansion_rate)
        self.linear2 = nn.Linear(input_dim * expansion_rate, output_dim)
    
    def forward(self, x):
        identity = x
        # print(x.shape)
        x = self.linear(x)
        x = self.silu(x)
        x = self.layer_norm(x)    
        x = self.linear2(x)
        # print(x.shape)
        x += identity
        
        return x


# %%


class V2AMapperMLP(nn.Module):
    """
    将(64,512)的clip特征先池化到(1,512),
    再映射到(1,512).
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super().__init__()
        # 可以先做一个简单的线性层, 或者堆叠多层
        self.pooling = nn.AdaptiveAvgPool2d((1, input_dim))  
        # pooling后, shape变成 (1, input_dim)

        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim * 2, hidden_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        self.v2a = nn.ModuleList([
            V2AMapper(input_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, output_dim, expansion_rate=1)
        ])
    def forward(self, x):
        # x: (batch_size, 64, 512)
        # 先把 shape (B,64,512) pooling 到 (B,1,512)
        # 这里可以用简单的mean替代，也可以用AdaptiveAvgPool2d
        pooled = x.mean(dim=1)  # (B,512)

        # 送入多层感知机映射到(512)
        # out = self.v2a(pooled)  # (B,512)
        for layer in self.v2a:
            pooled = layer(pooled)
        out = pooled
        return out




# %%
class V2ABlock(nn.Module):
    def __init__(self, dim, expansion_rate=2, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expansion_rate
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x + identity)
        return x

class V2AMapperMLPImproved(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.blocks = nn.Sequential(
            V2ABlock(input_dim, expansion_rate=2),
            V2ABlock(input_dim, expansion_rate=2),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, 512)
        return self.blocks(pooled)


# %%
def train_model(model, train_loader, criterion, optimizer, scaler):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0
    for i, (clip_feat, clap_feat) in enumerate(train_loader):
        # clip_feat = clip_feat.to(device)
        # clap_feat = clap_feat.to(device)
        # clip_feat = clip_feat.to(torch.float32).to(device)
        # clap_feat = clap_feat.to(torch.float32).to(device)
        clip_feat = clip_feat.float().to(device)
        clap_feat = clap_feat.float().to(device)
        if i == 0:
            print(f"[Debug] clip_feat device = {clip_feat.device}, clap_feat device = {clap_feat.device}")
        


        # 前向传播
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(clip_feat)  
            loss = criterion(outputs, clap_feat.squeeze(1)) 

        # 反向传播
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16


        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # del clip_feat, clap_feat, outputs, loss
        # torch.cuda.empty_cache()

    return total_loss / len(train_loader)

    

# %%

def validate_model(model, val_loader, criterion, optimizer):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0
    for i, (clip_feat, clap_feat) in enumerate(val_loader):
        clip_feat = clip_feat.to(device)
        clap_feat = clap_feat.to(device)

        with torch.no_grad():
            outputs = model(clip_feat)  
            loss = criterion(outputs, clap_feat.squeeze(1)) 

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar
        # del clip_feat, clap_feat, outputs, loss
        # torch.cuda.empty_cache()
    
    batch_bar.close()
    return total_loss / len(val_loader)

# %%
def train(model, train_loader, val_loader, criterion, optimizer,scaler, scheduler, ckpt_dir, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_model(model, train_loader, criterion, optimizer, scaler)
        val_loss = validate_model(model, val_loader, criterion, optimizer)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if True:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr
        })

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir + "best_model.pth")
            print("Model saved!")
        else:
            print("Validation loss did not improve, model not saved.")
            
        torch.save(model.state_dict(), ckpt_dir + "last_model.pth")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineLoss(nn.Module):
    def __init__(self, margin=0.8):  # 建议 margin 设得稍高一点
        super(CosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        # output: (B, 512)
        # target: (B, 1, 512) or (B, 512)
        if target.ndim == 3:
            target = target.squeeze(1)
        cos_sim = F.cosine_similarity(output, target, dim=1)  # (B,)
        loss = torch.mean(torch.clamp(self.margin - cos_sim, min=0))
        return loss


# %%
print(device)

# %%
# show me the mode summary 
from torchsummary import summary
# model = V2AMapperMLP(input_dim=512, hidden_dim=512, output_dim=512).to(device)
model = V2AMapperMLPImproved().to(device)

summary(model, input_size=(64, 512))

# %%
# Use wandb? Resume Training?
USE_WANDB = True

RESUME_LOGGING = False # Set this to true if you are resuming training from a previous run

# Create your wandb run

run_name = 'mlp-vggsound-simplefied_mapper_mse' # Give your run a name, this will be used to identify the run in wandb

# If you are resuming an old run
if USE_WANDB:

    wandb.login(key="8475199febe13b3465c7d5e4a595bba7422c14fc") #TODO

    if RESUME_LOGGING:
        run = wandb.init(
            id     = "", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs
            project = "v2amapper", ### Project should be created in your wandb
            settings = wandb.Settings(_service_wait=300)
        )


    else:
        run = wandb.init(
            name    = run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True, ### Allows reinitalizing runs when you re-run this cell
            project = "v2amapper", ### Project should be created in your wandb account
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)
        ### Save it in a txt file
        arch_file   = open("model_arch.txt", "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### log it in your wandb run with wandb.save()
        wandb.save('model_arch.txt')

# %%


# %%
lr = 0.001
epochs = 120
# model = V2AMapperMLP(input_dim=512, hidden_dim=1024, output_dim=512).to(device)
criterion = nn.MSELoss()
# criterion = CosineLoss(margin= 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000005)
scaler = torch.amp.GradScaler(enabled=True)

# ckpt_dir = "checkpoints"

train(model, train_loader, val_loader, criterion, optimizer,scaler, scheduler, ckpt_dir = "ckpts/", num_epochs=epochs)

# %%



