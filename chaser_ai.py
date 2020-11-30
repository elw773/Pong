"""
Team Name: DonkeyPong
"""

from math import *

# Globals
ai_exists = False
my_ai = None


def pong_ai(my_data, opponent_data, ball_data, table_size):
    global ai_exists, my_ai, hits

    # Our AI only needs to be created once.
    if not ai_exists:
        my_ai = Paddle_AI(my_data, opponent_data, ball_data, table_size)
        ai_exists = True

    paddle_pos = my_data.pos[1] + my_data.size[1] / 2  # Paddle position.
    ball_v = my_ai.ball_velocity(ball_data)  # Ball velocity.

    if ball_v[0] == 0:
        return "skip"

    if my_ai.ball_to_me(my_data, ball_data):
        go_to = my_ai.chase_ball(my_data, ball_data, ball_v, table_size)
    else:  # Defensive mode.
        go_to = my_ai.chase_ball(opponent_data, ball_data, ball_v, table_size)

        # If ball hits the other side way to high or too low, then go to the center.
        if go_to < (table_size[1] / 2 - my_data.size[1]) or go_to > (table_size[1] / 2 + my_data.size[1]):
            go_to = table_size[1] / 2

    my_ai.update_data(my_data, ball_data)

    if paddle_pos > go_to:
        return "up"
    elif paddle_pos < go_to:
        return "down"
    else:
        return "skip"


class Paddle_AI:
    """
    Models an artificially intelligent pong player.
    """

    def __init__(self, my_data, opponent_data, ball_data, table_size):
        """
        Initialize new Paddle AI self.
        """
        self.me = my_data
        self.enemy = opponent_data
        self.ball = ball_data
        self.table = table_size
        self.prev_ball_pos = [table_size[0] / 2, table_size[1] / 2]  # Ball starts off in the center.
        self.prev_dist = abs(my_data.pos[0] - self.prev_ball_pos[0])  # Previous distance to the ball.
        self.ball_v = (0, 0)  # Ball starts off in a stationary state.

    def update_data(self, my_data, ball_data):
        """
        Update data for self.
        """
        self.prev_ball_pos = ball_data.pos
        self.ball_v = self.ball_velocity(ball_data)
        self.prev_dist = abs(my_data.pos[0] - ball_data.pos[0])

    def ball_velocity(self, ball_data):
        """
        Return the velocity of the ball.
        """
        return ball_data.pos[0] - self.prev_ball_pos[0], ball_data.pos[1] - self.prev_ball_pos[1]

    def ball_to_me(self, my_data, ball_data):
        """
        Return whether or not the ball is coming towards the paddle.
        """
        current_dist = abs(my_data.pos[0] - ball_data.pos[0])

        if self.prev_dist > current_dist:
            return True
        else:
            return False

    def chase_ball(self, my_data, ball_data, ball_v, table_size):
        """
        Return a prediction (where the ball will hit).
        """
        hit_height = (ball_data.pos[1] + ball_v[1] * ((my_data.pos[0] - ball_data.pos[0])) / ball_v[0])
        bounces = abs(floor(hit_height / table_size[1]))

        if bounces % 2 == 0:  # Ball bounced even number of times.
            hit_height = hit_height % table_size[1]
        else:  # Ball bounced odd number of times.
            hit_height = table_size[1] - (hit_height % table_size[1])

        return hit_height