import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

train_data = "/home/ordn/Documents/ordn_projects/car_game_ai/training_data.txt"

from engine import SelectOutput

def clean_data(data, batch):
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

  for i,d in enumerate(_games['x']):
    if len(d) < batch:
      _games['x'].pop(i)
      _games['y'].pop(i)
      i-=1

  return _games, vocabs

batch     = 600
data      = open(train_data, 'r').read().splitlines()

class CarGameModelv2(nn.Module):

  def __init__(self,checkpoint_filename="car_ai_checkpointv2.pth"):
    self.fixed_batch = batch
    self.batch = torch.randint(1, self.fixed_batch, (1,)).item()
    # dataset handling...
    self._games, self.vocabs = clean_data(data, self.batch)

    super().__init__()
    self.initialize_game()
    self.checkpoint_filename = (checkpoint_filename)
    self.model = nn.Sequential(
      nn.Linear(33, 80),
      nn.LSTM(input_size=80,hidden_size=100,num_layers=2),
      SelectOutput(),nn.Linear(100,50),
      nn.LayerNorm(50),nn.Tanh(),nn.Linear(50,5))

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

  def initialize_game(self):
    self.mem = None
    self.fed = 0
    self.batch  = torch.randint(1, self.fixed_batch, (1,)).item()

  def get_traind(self):
    import random

    gix    = random.choice(list(range(0,len(self._games['x']))))
    xd, yd = torch.tensor(self._games['x'][gix], dtype=torch.float32), torch.tensor(self._games['y'][gix], dtype=torch.long)
    _tg    = torch.randint(0, max([1,(len(xd)-self.fixed_batch)]), (1,)).item()
    _tr    = torch.arange(_tg, _tg+self.fixed_batch)
    x, y   = xd[_tr], yd[_tr]

    return x,y

  def get_timed_traind(self, dur:tuple[int,int]=(0, 600)):
    if len(dur) != 2:
      raise ValueError(f"Invalid duration range. Ensure that dur is a tuple of two integers (start, end).")
    if dur[0] < 0 or dur[1] < 0 or dur[0] >= dur[1]:
      raise ValueError(f"Invalid duration range. Ensure that 0 <= dur[0]:{dur[0]} < dur[1]:{dur[1]}.")
    
    xtraind = []
    ytraind = []
    for i,d in enumerate(self._games['x']):
      if len(d) >= dur[1]:
        xtraind.append(d[dur[0]:dur[1]])
        ytraind.append(self._games['y'][i][dur[0]:dur[1]])

    ix = torch.randint(0, max([0,len(xtraind)]), (1,)).item()
    xtraind = [xtraind[ix]]
    ytraind = [ytraind[ix]]

    return torch.tensor(xtraind, dtype=torch.float32), torch.tensor(ytraind, dtype=torch.long)



  def get_deflt(self):
    return [self._games['x'][0][0]]

  def forward(self,x,y=None):
    logits = self.model(x)

    if y is not None:
      if y.ndim == 2:
        logits = logits.transpose(1,2) # transpose to [B, C, T] for cross_entropy
      loss = F.cross_entropy(logits,y)
    else:
      loss = None

    return logits, loss

  def fit(self,epochs=100):
    self.train()

    for epoch in range(epochs):
      self.batch = torch.randint(1, self.fixed_batch, (1,)).item() if epoch > (epochs*70) else self.batch
      xb,yb = self.get_traind()

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

  @torch.no_grad()
  def sample_game(self, x):
    x     = self.model[0](x)
    x, self.mem = self.model[1](x, self.mem) if self.mem is not None else self.model[1](x)

    for layer in self.model[3:]:
      x = layer(x)

    logits = x
    probs  = F.softmax(logits[-1],dim=0)
    ix     = torch.multinomial(probs, num_samples=1, replacement=True).item()
    return ix

  def save_checkpoint(self):
    checkpoint = {
      # Model weights
      "model_state_dict":self.state_dict(),
      "optimizer_state_dict":self.optim.state_dict(),
      "epochs_trained":self.epochs_trained,
      "vocabs":self.vocabs}

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