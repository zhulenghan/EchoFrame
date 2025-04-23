import torch
import os
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from vgg_dataset import *
from models import *
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def validate_model(model, val_loader, criterion, optimizer):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0
    for i, (clip_feat, clap_feat) in enumerate(val_loader):
        clip_feat = clip_feat.float().to(device)
        clap_feat = clap_feat.float().to(device)

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