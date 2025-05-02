import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import wandb
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt



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


# — Attention Block (unchanged) —
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# — Double Convolution block (unchanged) —
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

# — Wider Attention U-Net —
class AttUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3,
                 features=[128, 256, 512, 1024]):
        super().__init__()
        # Encoder path
        self.downs = nn.ModuleList()
        in_channels = n_channels
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.pool = nn.MaxPool2d(2)

        # Bottleneck is twice the last feature size
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder path with Attention
        self.ups = nn.ModuleList()
        self.attns = nn.ModuleList()
        rev_features = features[::-1]
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.attns.append(AttentionBlock(F_g=f, F_l=f, F_int=f//2))
            self.ups.append(DoubleConv(f*2, f))

        # Final 1×1 conv to get n_classes channels
        self.final_conv = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            up_trans = self.ups[idx]
            conv      = self.ups[idx+1]
            attn      = self.attns[idx//2]

            x = up_trans(x)
            skip = skips[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            skip = attn(g=x, x=skip)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final_conv(x)


# … later in your training script …

# instantiate the wider model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AttUNet(n_channels=1, n_classes=3,
                 features=[128,256,512,1024]).to(device)

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

def compute_ctr(lung_mask: np.ndarray, heart_mask: np.ndarray) -> float:
    """
    lung_mask, heart_mask: 2D boolean arrays of shape [H,W]
    returns cardiothoracic ratio = heart_width / chest_width
    """
    # chest bounds from lung mask
    ys, xs = np.where(lung_mask)
    chest_left, chest_right = xs.min(), xs.max()
    chest_width = chest_right - chest_left
    
    # heart bounds from heart mask
    ys_h, xs_h = np.where(heart_mask)
    heart_left, heart_right = xs_h.min(), xs_h.max()
    heart_width = heart_right - heart_left
    
    return heart_width / chest_width

def extract_ctrs(loader, model, device):
    model.eval()
    ctrs_gt, ctrs_pred = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            # predict
            logits = model(imgs)                  # [B,3,H,W]
            preds  = logits.argmax(dim=1).cpu().numpy()  # [B,H,W]
            gts    = masks.numpy()                      # [B,H,W]
            
            for pred, gt in zip(preds, gts):
                lung_pred  = (pred == 1)
                heart_pred = (pred == 2)
                lung_gt    = (gt   == 1)
                heart_gt   = (gt   == 2)
                
                ctrs_pred.append(compute_ctr(lung_pred, heart_pred))
                ctrs_gt.append(compute_ctr(lung_gt,   heart_gt))
    return np.array(ctrs_gt), np.array(ctrs_pred)

# — after training, on validation set —
val_ctrs_gt,  val_ctrs_pred  = extract_ctrs(val_loader,  model, device)
# — and on test set —
test_ctrs_gt, test_ctrs_pred = extract_ctrs(test_loader, model, device)

# compute summary metrics
val_mae  = np.mean(np.abs(val_ctrs_gt  - val_ctrs_pred))
test_mae = np.mean(np.abs(test_ctrs_gt - test_ctrs_pred))

print(f"Validation CTR MAE: {val_mae:.3f}, mean GT={val_ctrs_gt.mean():.3f}, mean Pred={val_ctrs_pred.mean():.3f}")
print(f"Test       CTR MAE: {test_mae:.3f}, mean GT={test_ctrs_gt.mean():.3f}, mean Pred={test_ctrs_pred.mean():.3f}")

# optionally log to wandb
import wandb
wandb.log({
    "val_ctr_mae":  val_mae,
    "val_ctr_gt":   val_ctrs_gt.mean(),
    "val_ctr_pred": val_ctrs_pred.mean(),
    "test_ctr_mae": test_mae,
    "test_ctr_gt":  test_ctrs_gt.mean(),
    "test_ctr_pred":test_ctrs_pred.mean(),
})


wandb.finish()
