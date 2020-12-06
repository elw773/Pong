# Pong
 AI for 2020 ESC180 Pong Competition
 
 [chaser_ai.py](chaser_ai.py) is the sample AI \
 [old_pong_ai.py](old_pong_ai.py) assumes perfect bounces to predict the ball's movements, and scores potential hitting 
 locations to select the best one. To reduce the time to return a move, long computations are split between consecutive 
 calls of pong_ai() \
 [pong_ai.py](pong_ai.py) improves on [old_pong_ai.py](old_pong_ai.py) by modeling the ball's motion the same way the 
 game does for increased accuracy and uses threading to reduce the time to return a move \
 [engine.py](engine.py) ws used for testing runs many games with no graphics or game loop delay
