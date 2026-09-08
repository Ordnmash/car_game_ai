import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

train_data = "/home/ordn/Documents/ordn_projects/car_game_ai/training_data.txt"

data = open(train_data, 'r').read().splitlines()

def clean_data(data):
  # this function handles the data of multiple games separated by '/'
  done_all = False
  _games   = {'x':[],'y':[]}

  while not done_all:
    idata    = []
    gdata    = {'x':[],'y':[]}
    for i,d in enumerate(data):
      if d != '/':
        idata.append(int(d))
      else:
        data = data[i+1:] # strip the collected data and strip the separator
        break
    # after break idata is now the whole each game massive list of int
    c  = 0
    gd = []
    for d in idata:
      gd.append(d)
      c+=1
      if c == 34:
        gdata['x'].append(gd[:33])
        gdata['y'].append(gd[33])
        c  = 0
        gd = []
      
    # now gdata == {'x':[xframe_1,xframe_2,...xframe_n],'y':[y1, y2, y3,...yn]} for each game
    _games['x'].append(gdata['x'])
    _games['y'].append(gdata['y'])
    vocabs = list(set(y for game in _games['y'] for y in game))
    if len(data) == 0:
      done_all = True

  return _games, vocabs

def get_deflt():
  return [_games['x'][0][0]]

def get_traind(d: dict, batch:int):
  import random

  gix    = random.choice(list(range(0,len(d['x']))))
  xd, yd = torch.tensor(d['x'][gix], dtype=torch.float32), torch.tensor(d['y'][gix], dtype=torch.long)
  _tg    = torch.randint(0, len(xd)-batch, (1,)).item()
  _tr    = torch.arange(_tg, _tg+batch)
  x, y   = xd[_tr], yd[_tr]

  return x,y

batch = 180
_games, vocabs = clean_data(data)

class SelectOutput(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    output = x[0]
    return output

  
class CarGameModel(nn.Module):

  def __init__(self,checkpoint_filename="car_ai_checkpoint.pth"):
    super().__init__()
    self.batch = batch
    self.checkpoint_filename = (checkpoint_filename)
    self.model = nn.Sequential(
      nn.LSTM(input_size=33,hidden_size=66,num_layers=2),
      SelectOutput(),nn.Linear(66,33),
      nn.LayerNorm(33),nn.Tanh(),nn.Linear(33,5))

    self.optim = optim.AdamW(self.model.parameters(),lr=1e-3)

    self.epochs_trained = 0

    parameter_count = sum(p.nelement() for p in self.parameters())
    print(
      f"Number of parameters: "
      f"{parameter_count}")

    if os.path.exists(self.checkpoint_filename):
      print("\nCheckpoint found.")
      self.load_checkpoint()
    else:
      print("\nNo checkpoint found...")
      print("Starting with a new model...")

  def forward(self,x,y=None):
    logits = self.model(x)

    if y is not None:
      loss = F.cross_entropy(logits,y)
    else:
      loss = None

    return logits, loss

  def fit(self,epochs=100):
    self.train()

    for epoch in range(epochs):
      xb,yb = get_traind(_games, batch)

      _, loss = self(xb,yb)

      self.optim.zero_grad(set_to_none=True)

      loss.backward()

      self.optim.step()
      self.epochs_trained += 1

      print_every = max(1,epochs // 10)
      if ((epoch + 1) % print_every == 0):
        print(
          f"Epoch: "
          f"{self.epochs_trained} "
          f"| Loss: "
          f"{loss.item():.4f}")

    self.save_checkpoint()

  def sample_game(self, x, ht=None):
    x, ht = self.model[0](x, ht)
    for layer in self.model[2:]:
      x = layer(x)
    logits = x
    probs  = F.softmax(logits[-1],dim=0)
    ix     = torch.multinomial(probs, num_samples=1, replacement=True).item()
    return ix, ht

  def save_checkpoint(self):
    checkpoint = {
      # Model weights
      "model_state_dict":self.state_dict(),
      "optimizer_state_dict":self.optim.state_dict(),
      "epochs_trained":self.epochs_trained,
      "vocabs":vocabs}

    torch.save(checkpoint,self.checkpoint_filename)

    print("\nCheckpoint saved:")
    print(self.checkpoint_filename)
    print(f"Total epochs trained: "
      f"{self.epochs_trained}")

  def load_checkpoint(self):
    checkpoint = torch.load(self.checkpoint_filename,weights_only=False,map_location="cpu")
    self.load_state_dict(checkpoint["model_state_dict"])
    self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
    self.epochs_trained = (checkpoint.get("epochs_trained",0))

    print("Model loaded successfully.")
    print("Optimizer loaded successfully.")
    print(
      f"Previous epochs trained: "
      f"{self.epochs_trained}")