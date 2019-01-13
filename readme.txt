基于模仿学习训练神经网络控制2048游戏
李蔚 516021910222

·模型'model666.h5'，自己的agent定义在agent.py里：

import keras
from keras.models import load_model
import time
model=load_model('model666.h5')

class MyAgent(Agent):
    def __init__(self, game, display=None):
        self.game = game
        self.display = display		
    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
    def step(self):
        x=np.array(self.game.board)
        x=np.log2(x+1)
        x=np.trunc(x)
        x = keras.utils.to_categorical(x,12)
        direction = int(model.predict(x.reshape(1,4,4,12),batch_size=128).argmax())              
        return direction
		
·使用时
    from game2048.game import Game
    game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
    from game2048.agents import MyAgent
    agent = MyAgent(game=game)
	agent.play()

·导入模型使用
    from keras.models import load_model
    model=load_model('model666.h5')
	...