# training scheme
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import os
from torch.cuda.amp import GradScaler

# Training step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               scaler: torch.cuda.amp.GradScaler) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # for mixed precision training
        with torch.cuda.amp.autocast():
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        if scaler.is_enabled() and any(torch.isinf(p.grad).any() or torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
          print("Detected inf or NaN in gradients. Unscaling and skipping step.")
          scaler.unscale_(optimizer)
          scaler.update()
          continue  # Skip this step
        scaler.step(optimizer)
        scaler.update()
      

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Validation step function
def validate_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: torch.device) -> Tuple[float, float]:
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_pred_logits = model(X)

            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

# Test step function
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Training function with early stopping, checkpointing, and mixed precision
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler = None,
          early_stopping_patience: int = 10,
          checkpoint_path: str = 'best_model.pth') -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           scaler=scaler)
        val_loss, val_acc = validate_step(model=model,
                                          dataloader=val_dataloader,
                                          loss_fn=loss_fn,
                                          device=device)
        if scheduler:
            scheduler.step()

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load the best model for testing
    model.load_state_dict(torch.load(checkpoint_path))

    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    return results





class stage1Config():
  def __init__(self):
    self.num_heads=1
    self.num_layers=1
    self.embed_dim=64
    self.block_size=56**2
    self.kernel_size=3
    self.padding=1
    self.in_channels=3

class stage2Config():
  def __init__(self):
    self.num_heads=3
    self.num_layers=2
    self.embed_dim=192
    self.block_size=784
    self.kernel_size=3
    self.padding=1
    self.in_channels=64

class stage3Config():
  def __init__(self):
    self.num_heads=6
    self.num_layers=10
    self.embed_dim=384
    self.block_size=196
    self.kernel_size=3
    self.padding=1
    self.in_channels=192

class CViTConfig():
  def __init__(self):   
    self.stage_1_config = stage1Config()    
    self.stage_2_config = stage2Config()   
    self.stage_3_config = stage3Config()
    self.final_embedsize = 384
    self.num_classes=1000




class Attention(nn.Module):
  def __init__(self, config, stage_three=False):
    super().__init__()
    self.stage_three = stage_three
    self.config = config
    self.Q = nn.Sequential(
        nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=config.kernel_size, stride=1, padding=config.padding, groups=config.embed_dim),
        nn.BatchNorm2d(config.embed_dim, eps=1e-6),
        nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=1, stride=1, padding=0),

    )
    self.K = nn.Sequential(
        nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=config.kernel_size, stride=2, padding=config.padding, groups=config.embed_dim),
        nn.BatchNorm2d(config.embed_dim, eps=1e-6),
        nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=1, stride=1, padding=0),
    )
    self.V = nn.Sequential(
        nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=config.kernel_size, stride=2, padding=config.padding, groups=config.embed_dim),
        nn.BatchNorm2d(config.embed_dim, eps=1e-6),
        nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=1, stride=1, padding=0),
    )

  def forward(self, x):
    # If stage 3, x -> B, T+1, C (with class token)
    if self.stage_three==True:
        cls_token = x[:, 0, :].unsqueeze(1)  # Isolate class token
        x = x[:, 1:, :]  # Remove class token from the rest
        x = x.permute(0, 2, 1)  # B, C, T
        B,C,T = x.shape
        # Apply convolutions and permute back
        Q = self.Q(x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)).view(B,C,T).permute(0, 2, 1)  # B, T, C
        K = self.K(x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)).view(B,C,-1).permute(0, 2, 1)  # B, T/2, C
        V = self.V(x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)).view(B,C,-1).permute(0, 2, 1)  # B, T/2, C

        # Reinsert the class token
        Q = torch.cat([cls_token, Q], dim=1)
        K = torch.cat([cls_token, K], dim=1)
        V = torch.cat([cls_token, V], dim=1)
    else:
        x = x.permute(0, 2, 1)  # B, C, T , convolution expects, B,C,t,t
        B,C,T = x.shape
        Q = self.Q(x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)).view(B,C,T).permute(0, 2, 1)  # B, T, C
        K = self.K(x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)).view(B,C,-1).permute(0, 2, 1)  # B, T/2, C
        V = self.V(x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)).view(B,C,-1).permute(0, 2, 1)  # B, T/2, C

    # Ensure embedding dimension is divisible by num_heads
    assert Q.shape[-1] % self.config.num_heads == 0, "Embed_dim must be divisible by num_heads"

    B, T, C = Q.shape
    T_k = K.shape[1]

    # Reshape for multihead attention: Q, K, V -> B, nh, T, Hs
    Q = Q.view(B, T, self.config.num_heads, C // self.config.num_heads).transpose(1, 2)
    K = K.view(B, T_k, self.config.num_heads, C // self.config.num_heads).transpose(1, 2)
    V = V.view(B, T_k, self.config.num_heads, C // self.config.num_heads).transpose(1, 2)

    # Apply scaled dot-product attention
    attn = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=False)

    # Reshape back to B, T, C
    attn = attn.transpose(1, 2).contiguous().view(B, T, C)   
    return attn


class EncoderLayer(nn.Module):
  def __init__(self, config, stage_three):
    super().__init__()
    self.stage_three = stage_three
    self.config = config
    self.ln_1 = nn.LayerNorm(self.config.embed_dim, bias=True, eps=1e-6)
    self.ln_2 = nn.LayerNorm(self.config.embed_dim, bias=True, eps=1e-6)
    self.self_attention = Attention(config, self.stage_three)
    self.mlp = nn.Sequential(
        nn.Linear(config.embed_dim, 4*config.embed_dim),
        nn.GELU(),
        nn.Linear(4*config.embed_dim, config.embed_dim)
    )
  def forward(self, x):
    #x is of shape B, T, C
    x = x + self.self_attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))

    return x



class EncoderLayers(nn.Module):
  def __init__(self, config, stage_three=False):
    super().__init__()
    self.stage_three = stage_three
    self.config = config
    self.ln = nn.LayerNorm(self.config.embed_dim, bias=True, eps=1e-6)
    self.layers = nn.ModuleDict({f"encoder_layer_{_}":EncoderLayer(config, self.stage_three) for _ in range(self.config.num_layers)})
  def forward(self, x):
    for layer in self.layers.values():
      x = layer(x)
    x = self.ln(x) 
    return x


class Block(nn.Module):
  def __init__(self,
               config=None,
               stage_three=False):
    super().__init__()
    self.config = config
    self.stage_three = stage_three

    if config is None:
      self.conv_embed = nn.Conv2d(3,64, kernel_size=7, stride=4, padding=2)
      self.config = CViTConfig().stage_1_config
    else:
      self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
      self.conv_embed = nn.Conv2d(self.config.in_channels,self.config.embed_dim, kernel_size=self.config.kernel_size, stride=2, padding=self.config.padding)
    self.encoder = EncoderLayers(self.config, self.stage_three)
  def forward(self, x):
    if self.config.in_channels==3:
      #B,Ch,H,W->B,C,txt
      x = self.conv_embed(x)
      
    else:
      #B,T,C->B,C,txt
      x = x.permute(0, 2, 1)
      x = x.view(x.shape[0], x.shape[1], int(x.shape[-1]**0.5),-1)
      x = self.conv_embed(x)

    #B,C,txt ->B,C,T
    x = x.flatten(2)
 
    #B,C,T -> B,T,C
    x = x.transpose(1, 2).contiguous()
    if self.stage_three==True:
      x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.encoder(x)
    return x
class CViTHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.head = nn.Linear(config.final_embedsize,config.num_classes)
  def forward(self, x):
    x = self.head(x[:,0,:])

    return x

class CViT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.stage_1 = Block()
    self.stage_2 = Block(self.config.stage_2_config)
    self.stage_3 = Block(self.config.stage_3_config, True)
    self.heads = CViTHead(self.config)
  def forward(self, x):
    x = self.stage_1(x)   
    x = self.stage_2(x)
    x = self.stage_3(x)    
    x = self.heads(x)   
    return x

  def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


cvit = CViT(CViTConfig())
cvit(torch.randn(32, 3, 224, 224)).shape
