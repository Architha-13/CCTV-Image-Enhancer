import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------
# Residual Dense Block
# -----------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


# -----------------------------------
# RRDB Block
# -----------------------------------
class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


# -----------------------------------
# Full RRDBNet (MATCHES RealESRGAN)
# -----------------------------------
class RRDBNet(nn.Module):
    def __init__(self, scale=4, nf=64, nb=23, gc=32):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # IMPORTANT: names must match checkpoint
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


# -----------------------------------
# Load Model
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RRDBNet().to(device)

state = torch.load("models/net_g_15000.pth", map_location=device)

# Extract correct weight dictionary
if isinstance(state, dict):
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    elif "model" in state:
        state = state["model"]

model.load_state_dict(state, strict=True)
model.eval()


# -----------------------------------
# Inference Function
# -----------------------------------
def upscale_image(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)

    return output