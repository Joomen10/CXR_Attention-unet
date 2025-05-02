import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import os
import wandb


class CXRSegmentationDataset(Dataset):
    def __init__(self, img_dir, lungs_dir, heart_dir,
                 size=(512,512),
                 augment=False):
        """
        img_dir   : folder of resized grayscale JPEGs of original X‑rays
        lungs_dir : folder of resized lungs masks (0/255 PNGs)
        heart_dir : folder of resized heart masks (0/255 PNGs)
        size      : (width, height) to resize to
        augment   : whether to apply random flips/rotations
        """
        # collect IDs from the .jpg files in img_dir
        self.ids = [os.path.splitext(f)[0]
                    for f in os.listdir(img_dir)
                    if f.lower().endswith(".jpg")]
        print(f"Found {len(self.ids)} samples in {img_dir}")
        
        self.img_dir   = img_dir
        self.lungs_dir = lungs_dir
        self.heart_dir = heart_dir

        # image transformations
        self.tf_img = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),                         # [1,H,W], floats in [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])    # adjust to your data
        ])
        # mask resizing (nearest to preserve labels)
        self.tf_mask = transforms.Resize(size,
                                         interpolation=transforms.InterpolationMode.NEAREST)

        # optional augmentations
        self.aug = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]) if augment else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]

        img_path   = os.path.join(self.img_dir,   f"{_id}.jpg")
        lung_path  = os.path.join(self.lungs_dir, f"{_id}_lungs.png")
        heart_path = os.path.join(self.heart_dir, f"{_id}_heart.png")

        img    = Image.open(img_path).convert("L")
        mask_l = Image.open(lung_path)
        mask_h = Image.open(heart_path)

        # Apply same random augmentation to all three
        if self.aug:
            seed = np.random.randint(0, 1_000_000)
            torch.manual_seed(seed)
            img    = self.aug(img)
            torch.manual_seed(seed)
            mask_l = self.aug(mask_l)
            torch.manual_seed(seed)
            mask_h = self.aug(mask_h)

        # Resize
        img    = self.tf_img(img)         # tensor [1,H,W]
        mask_l = self.tf_mask(mask_l)     # PIL image resized
        mask_h = self.tf_mask(mask_h)

        # Convert masks to tensors [1,H,W] uint8
        mask_l = transforms.PILToTensor()(mask_l)
        mask_h = transforms.PILToTensor()(mask_h)

        # Build a single multi-class mask: 0=BG,1=Lung,2=Heart
        ml = (mask_l.squeeze(0) // 255).to(torch.uint8)
        mh = (mask_h.squeeze(0) // 255).to(torch.uint8)
        mask = torch.zeros_like(ml, dtype=torch.uint8)
        mask[ml == 1] = 1
        mask[mh == 1] = 2

        return img, mask.long()  # img: [1,H,W], mask: [H,W] ints in {0,1,2}



import torch
import torch.nn as nn
import torch.nn.functional as F

# — Attention Block —
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # gating signal F_g, skip connection F_l
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# — Double Convolution block —
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

# — Attention U‑Net —
class AttUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, features=[64,128,256,512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        for f_in, f_out in zip([n_channels]+features, features):
            self.downs.append(DoubleConv(f_in, f_out))
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # Decoder with Attention
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        rev_features = features[::-1]
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.attentions.append(AttentionBlock(F_g=f, F_l=f, F_int=f//2))
            self.ups.append(DoubleConv(f*2, f))
        # Final conv
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            up_trans = self.ups[idx]
            attn     = self.attentions[idx//2]
            conv     = self.ups[idx+1]
            x = up_trans(x)
            skip = skip_connections[idx//2]
            # crop if needed to match sizes
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            # attention & concatenation
            skip = attn(g=x, x=skip)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return self.final_conv(x)



import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import wandb
import matplotlib.pyplot as plt
from PIL import Image

# — your Dataset & Model imports —
# from datasets import CXRSegmentationDataset
# from models import AttUNet

# 0) W&B setup
# 0) set your W&B key in‑script
os.environ["WANDB_API_KEY"] = "9b3cc6b608bb679c5cc822e2e256c754cc777ee0"
wandb.login(key=os.environ["WANDB_API_KEY"], force=True)
wandb.init(
    project="cxr-segmentation",
    name="attunet-3way-split",
    config={
        "epochs": 30,
        "batch_size": 8,
        "lr": 1e-4,
        "input_size": [512,512],
        "n_classes": 3
    }
)

# 1) build full dataset
full_ds = CXRSegmentationDataset(
    img_dir   = "processed_images",
    lungs_dir = "processed_masks/lungs",
    heart_dir = "processed_masks/heart",
    size      = (512,512),
    augment   = True
)

# 2) split into train / val / test (70/15/15)
N = len(full_ds)
n_train = int(0.8 * N)
n_val   = int(0.15 * N)
n_test  = N - n_train - n_val
train_ds, val_ds, test_ds = random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=wandb.config.batch_size,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=wandb.config.batch_size,
                          shuffle=False, num_workers=0)

# 3) model, optimizer, loss
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = AttUNet(n_channels=1, n_classes=wandb.config.n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
criterion = CrossEntropyLoss()

# 4) watch model
wandb.watch(model, log="all", log_freq=100)

# 5) training loop with val and image logging
for epoch in range(wandb.config.epochs):
    # — train —
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # — validate —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    # — log scalars —
    wandb.log({
        "epoch":      epoch+1,
        "train_loss": train_loss,
        "val_loss":   val_loss
    })

        # — random sample of 4 different val‐set indices —
    examples = []
    for _ in range(4):
        idx = random.randrange(len(val_ds))       # pick a random example
        img_t, mask_gt = val_ds[idx]             # img_t: [1,H,W], mask_gt: [H,W]
        
        # run through model
        img = img_t.unsqueeze(0).to(device)      # [1,1,H,W]
        with torch.no_grad():
            pred = model(img).argmax(dim=1)[0].cpu().numpy()  # [H,W]
        
        bg = img_t[0].cpu().numpy()              # [H,W] grayscale
        gt = mask_gt.cpu().numpy()               # [H,W]
        
        # build overlay figure
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(bg, cmap="gray")
        ax[0].imshow((gt==1), cmap="Reds",   alpha=0.3)
        ax[0].imshow((gt==2), cmap="Greens", alpha=0.3)
        ax[0].set_title("Ground Truth"); ax[0].axis("off")
    
        ax[1].imshow(bg, cmap="gray")
        ax[1].imshow((pred==1), cmap="Reds",   alpha=0.3)
        ax[1].imshow((pred==2), cmap="Greens", alpha=0.3)
        ax[1].set_title("Prediction"); ax[1].axis("off")
        plt.tight_layout()
    
        examples.append(wandb.Image(fig, caption=f"Val idx={idx}"))
        plt.close(fig)

    # log those 4 random validation examples as a single step
    wandb.log({"val_examples": examples, "epoch": epoch+1})
    

    # — log example overlays on validation set —
    # # pick first batch
    # imgs, masks = next(iter(val_loader))
    # imgs = imgs.to(device)
    # with torch.no_grad():
    #     preds = model(imgs).argmax(dim=1).cpu().numpy()  # [B,H,W]
    # imgs_np = imgs.cpu().numpy()[:,0]                  # [B,H,W]
    # masks_np = masks.numpy()                           # [B,H,W]

    # # build a list of wandb.Image objects
    # examples = []
    # for i in range(min(4, imgs_np.shape[0])):
    #     bg = imgs_np[i]
    #     pred = preds[i]
    #     gt   = masks_np[i]

    #     # overlay pred & gt
    #     fig, ax = plt.subplots(1,2, figsize=(8,4))
    #     ax[0].imshow(bg, cmap="gray")
    #     ax[0].imshow((gt==1).astype(float), cmap="Reds",   alpha=0.3)
    #     ax[0].imshow((gt==2).astype(float), cmap="Greens", alpha=0.3)
    #     ax[0].set_title("Ground Truth"); ax[0].axis("off")
    #     ax[1].imshow(bg, cmap="gray")
    #     ax[1].imshow((pred==1).astype(float), cmap="Reds",   alpha=0.3)
    #     ax[1].imshow((pred==2).astype(float), cmap="Greens", alpha=0.3)
    #     ax[1].set_title("Prediction");  ax[1].axis("off")
    #     plt.tight_layout()

    #     examples.append(wandb.Image(fig, caption=f"Val sample {i}"))
    #     plt.close(fig)

    # wandb.log({"val_examples": examples})

# 6) test‐time evaluation (just compute test loss & IoU)
model.eval()
test_loss = 0.0
all_preds, all_targs = [], []
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        test_loss += criterion(logits, masks).item() * imgs.size(0)
        pred   = logits.argmax(dim=1).cpu().numpy().reshape(-1)
        targ   = masks.cpu().numpy().reshape(-1)
        all_preds.append(pred); all_targs.append(targ)
test_loss /= len(test_loader.dataset)
all_preds = np.concatenate(all_preds)
all_targs = np.concatenate(all_targs)

from sklearn.metrics import jaccard_score
iou_lung  = jaccard_score(all_targs==1, all_preds==1)
iou_heart = jaccard_score(all_targs==2, all_preds==2)

wandb.log({
    "test_loss":  test_loss,
    "iou_lung":   iou_lung,
    "iou_heart":  iou_heart
})

# 7) save final model
torch.save(model.state_dict(), "attunet_final.pt")
wandb.save("attunet_final.pt")
wandb.finish()
