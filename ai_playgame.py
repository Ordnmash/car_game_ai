from main import Game
#from engine import CarGameModel as model1
from carAI_v2 import CarGameModelv2 as model2

#ai = model1("/home/ordn/Documents/ordn_projects/car_game_ai/car_ai_checkpoint.pth")
ai = model2()
gameplay = Game(ai)
gameplay.initialize_game()
gameplay.run_game_loop()