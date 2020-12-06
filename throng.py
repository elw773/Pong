# Ted Pinkerton and Ewan Wai

import math
import numpy as np

max_angle = 45
math_pi = math.pi
inf = math.inf
paddle_bounce = 1.2
max_loop = 400

import threading
from timeit import default_timer as timer
import time
import numpy


def move_to_paddle_x(paddle_x, ball_x, ball_y, ball_dx, ball_dy, direction, move_factor):
    """Calculates the position and velocity of the ball when it reaches the paddle_x
        Mutates the ball_pos
    """
    n = 0
    while (paddle_x - ball_x) * direction > 0:
        n += 1
        if (n > max_loop):
            if debug: print("Max loop moving to paddle")
            break
        wall_y = -1
        if ball_dy > 0:
            wall_y = TABLE_SIZE[1] + 1

        ticks_to_paddle = -int((paddle_x - ball_x) // (-ball_dx * move_factor))
        ticks_to_wall = -int((wall_y - ball_y) // (-ball_dy * move_factor))  # ceiling computation

        if ticks_to_paddle < ticks_to_wall:
            ball_x += ball_dx * ticks_to_paddle * move_factor
            ball_y += ball_dy * ticks_to_paddle * move_factor
        else:
            ball_x += ball_dx * move_factor * ticks_to_wall
            ball_y += ball_dy * move_factor * ticks_to_wall
            # n += 1

            d = -((ball_y - wall_y) // (ball_dy * -0.1 * move_factor))
            ball_y -= (2 * d) * ball_dy * 0.1 * move_factor

            ball_dy = -ball_dy

    return (ball_x, ball_y), (ball_dx, ball_dy)


def move_to_paddle(hitting_paddle_y, ball_x, ball_y, ball_dx, ball_dy, paddle_size, table_size,
                   move_factor):  # makes the ball actually hit the paddle
    if ball_y < hitting_paddle_y - paddle_size[1] / 2:  # if under paddle when at paddle level
        if ball_dy > 0:
            ticks_to_paddle = -int((hitting_paddle_y - paddle_size[1] / 2 - ball_y) // (-ball_dy * move_factor))
            ball_x += ball_dx * move_factor * ticks_to_paddle
            ball_y += ball_dy * move_factor * ticks_to_paddle
        else:  # ball will miss paddle
            return False
    elif ball_y > hitting_paddle_y + paddle_size[1] / 2:  # if above paddle when at paddle level
        if ball_dy < 0:
            ticks_to_paddle = -int((hitting_paddle_y + paddle_size[1] / 2 - ball_y) // (-ball_dy * move_factor))
            ball_x += ball_dx * move_factor * ticks_to_paddle
            ball_y += ball_dy * move_factor * ticks_to_paddle
        else:  # ball will miss paddle
            return False

    if ball_x <= -paddle_size[0] or ball_x >= table_size[0] + paddle_size[0]:  # if ball passed paddle
        return False

    return ball_x, ball_y


def do_hit(paddle_x, paddle_y, ball_x, ball_y, ball_dx, ball_dy, paddle_size, ball_size, table_size, direction,
           move_factor, hit=False):
    e = -((ball_x - paddle_x) // (ball_dx * -0.1 * move_factor))

    if ball_dy > 0:
        wall_y = table_size[1] + 1
        paddle_y_bound = paddle_y - paddle_size[1] / 2
    else:
        wall_y = -1
        paddle_y_bound = paddle_y + paddle_size[1] / 2
    d = -((ball_y - wall_y) // (ball_dy * -0.1 * move_factor))
    f = -((ball_y - paddle_y_bound) // (ball_dy * -0.1 * move_factor))
    if f > 0:
        e = min(e, f)
    if d > 0:
        e = max(e, d)

    # ball_x -= ball_dx * 0.1 * move_factor * e
    # ball_y -= ball_dy * 0.1 * move_factor * e
    c = 0
    while (ball_y > table_size[1] or ball_y < 0) or \
            (paddle_x - (paddle_size[0] if direction < 0 else 0) < ball_x < paddle_x + (
                    paddle_size[0] if direction > 0 else 0) and
             paddle_y - paddle_size[1] / 2 < ball_y < paddle_y + paddle_size[1] / 2):
        c += 1
        ball_x -= ball_dx * 0.1 * move_factor
        ball_y -= ball_dy * 0.1 * move_factor
        if c > max_loop:
            if debug: print("Max loop adjusting before hit")
            break

    # c = e

    rel_dist_from_c = (ball_y - paddle_y) / (paddle_size[1] - ball_size[1])
    rel_dist_from_c = min(0.5, rel_dist_from_c)
    rel_dist_from_c = max(-0.5, rel_dist_from_c)

    theta = direction * rel_dist_from_c * 45 * math_pi / 180

    ball_dx = math.cos(theta) * ball_dx - math.sin(theta) * ball_dy
    ball_dy = math.sin(theta) * ball_dx + math.cos(theta) * ball_dy

    ball_dx = -ball_dx

    ball_dx = math.cos(-theta) * ball_dx - math.sin(-theta) * ball_dy
    ball_dy = math.cos(-theta) * ball_dy + math.sin(-theta) * ball_dx

    # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
    if ball_dx * -direction < 1:  # ball is not traveling (a) away from paddle (b) at a sufficient speed
        if ball_dx ** 2 + ball_dy ** 2 < 1:
            return (None, None), (None, None)
        ball_dy = (ball_dy / abs(ball_dy)) * math.sqrt(
            ball_dx ** 2 + ball_dy ** 2 - 1)  # transform y velocity so as to maintain the speed
        ball_dx = direction * -1  # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to increase by *1.2

    if not hit:
        ball_dx *= paddle_bounce
        ball_dy *= paddle_bounce

    while c > 0 or (paddle_x - (paddle_size[0] if direction < 0 else 0) < ball_x < paddle_x + (
            paddle_size[0] if direction > 0 else 0) and
                    paddle_y - paddle_size[1] / 2 < ball_x < paddle_y + paddle_size[
                        1] / 2):
        ball_x += ball_dx * 0.1 * move_factor
        ball_y += ball_dy * 0.1 * move_factor
        c -= 1
        if c < -max_loop:
            if debug: print("Max loop adjusting after hit")
            break

    e = -((ball_x - paddle_x) // (ball_dx * -0.1 * move_factor))
    c = max(e, c)
    # ball_x += ball_dx * 0.1 * move_factor * c
    # ball_y += ball_dy * 0.1 * move_factor * c

    return (ball_x, ball_y), (ball_dx, ball_dy)


def hit_paddle(paddle_x, paddle_y, ball_x, ball_y, ball_dx, ball_dy, paddle_size, ball_size, table_size, direction):
    """Calculates the ball's position and velocity after hitting the paddle, assuming it is touching the paddle at the moment"""
    hit = False
    n = 0
    while not 0 < ball_x < table_size[0]:
        n += 1
        if n > 200:
            if debug: print("Max loop while trying to hit paddle")
            return (None, None), (None, None)
        g = int((ball_dx ** 2 + ball_dy ** 2) ** .5)
        move_factor = 1. / g if g > 0 else 1.0
        if (paddle_x - (paddle_size[0] if direction < 0 else 0) < ball_x < paddle_x + (
                paddle_size[0] if direction > 0 else 0) and
                paddle_y - paddle_size[1] / 2 < ball_y < paddle_y + paddle_size[
                    1] / 2):
            (ball_x, ball_y), (ball_dx, ball_dy) = do_hit(paddle_x, paddle_y, ball_x, ball_y, ball_dx, ball_dy,
                                                          paddle_size, ball_size, table_size, direction, move_factor,
                                                          hit)
            if ball_x is None:
                return (None, None), (None, None)
            hit = True
        else:
            ball_x += ball_dx * move_factor
            ball_y += ball_dy * move_factor

    return (ball_x, ball_y), (ball_dx, ball_dy)


def calc_hit(hitting_paddle_x, hitting_paddle_y, next_paddle_x, pre_hit_ball_x, pre_hit_ball_y, pre_hit_ball_dx,
             pre_hit_ball_dy, direction, move_factor):
    hit_pos = move_to_paddle(hitting_paddle_y, pre_hit_ball_x, pre_hit_ball_y, pre_hit_ball_dx, pre_hit_ball_dy,
                             PADDLE_SIZE, TABLE_SIZE, move_factor)
    if not hit_pos:
        return  # ball misses paddle
    else:
        (ball_x, ball_y), (ball_dx, ball_dy) = hit_paddle(hitting_paddle_x, hitting_paddle_y, hit_pos[0], hit_pos[1],
                                                          pre_hit_ball_dx, pre_hit_ball_dy,
                                                          PADDLE_SIZE, BALL_SIZE, TABLE_SIZE, direction)

        if ball_x is None:
            return

        g = int((ball_dx ** 2 + ball_dy ** 2) ** .5)
        move_factor = 1. / g if g > 0 else 1.0
        (ball_x, ball_y), (ball_dx, ball_dy) = move_to_paddle_x(next_paddle_x, ball_x, ball_y, ball_dx, ball_dy,
                                                                -direction, move_factor)

        return hitting_paddle_y, ball_x, ball_y, ball_dx, ball_dy


def calc_bounds(hit_y):
    upper_end = int(hit_y + PADDLE_SIZE[1] / 2) + 10.5
    upper_limit = TABLE_SIZE[1] - PADDLE_SIZE[1] / 2 + BALL_SIZE[1]

    lower_end = int(hit_y - PADDLE_SIZE[1] / 2) - 10.5
    lower_limit = PADDLE_SIZE[1] / 2 - BALL_SIZE[1]

    upper_bound = min(upper_end, upper_limit)
    lower_bound = max(lower_end, lower_limit)
    return lower_bound, upper_bound


def process_hits():
    global goal_paddle_hit, goal_paddle_defend
    while True:
        if calculating == 1 and calculated > 1:  # if we are searching for a hit and we calculated hits
            best_y, best_score = -1, -inf
            lowest_reach = paddle_pos[1] - t_to_hit * PADDLE_VEL
            highest_reach = paddle_pos[1] + t_to_hit * PADDLE_VEL
            for hit in hits:
                if lowest_reach <= hit[0] <= highest_reach:
                    t_to_them = t_to_hit + (hit[1] - paddle_pos[0]) // hit[3]
                    t_to_intercept = (abs(hit[2] - other_paddle_pos[1]) - PADDLE_SIZE[1] / 2) // PADDLE_VEL
                    score = t_to_intercept - t_to_them
                    if t_to_intercept > t_to_them:  # if we basically win
                        score *= 10e5
                    else:
                        #if calculated > 2:  # if i calculated returns
                         #   score = -their_scores[hit[0]]
                        #else:
                        score *= hit[3]

                    if score > best_score:
                        best_y, best_score = hit[0], score
            if more_debug: print("Calculated best hit")

            if best_y > -1:
                goal_paddle_hit = best_y
        if calculating == -1 and calculated == -1:
            g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
            move_factor = 1. / g if g > 0 else 1.0

            pre_hit_ball_pos, pre_hit_ball_vel = move_to_paddle_x(other_paddle_pos[0], ball_pos[0], ball_pos[1],
                                                                  ball_vel[0], ball_vel[1], direction, move_factor)

            prediction = calc_hit(other_paddle_pos[0], other_paddle_pos[1], paddle_pos[0],
                                  pre_hit_ball_pos[0], pre_hit_ball_pos[1], pre_hit_ball_vel[0],
                                  pre_hit_ball_vel[1], direction, move_factor)

            if not prediction is None:
                goal_paddle_defend = prediction[2]

            if t_to_hit * PADDLE_VEL > PADDLE_SIZE[1] / 4:
                sum, sum_of_weights = 0, 0
                lowest_reach = other_paddle_pos[1] - t_to_hit * PADDLE_VEL
                highest_reach = other_paddle_pos[1] + t_to_hit * PADDLE_VEL
                for hit in hits:
                    if lowest_reach <= hit[0] <= highest_reach:
                        t_to_me = t_to_hit + (hit[1] - other_paddle_pos[0]) // hit[3]
                        t_to_intercept = (abs(hit[2] - paddle_pos[1]) - PADDLE_SIZE[1] / 2) // PADDLE_VEL
                        weight = hit[3]
                        if weight <= 0:
                            weight = 0
                        sum += hit[0] * weight
                        sum_of_weights += weight
                if sum_of_weights != 0:
                    goal_paddle_defend = ((sum / sum_of_weights) + goal_paddle_defend) / 2
            if more_debug: print("Predicted return")


def away():
    global goal_paddle_defend, calculated, hits
    calculated = 0
    if debug: print("Predicting")

    # go to where the ball will hit
    g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
    move_factor = 1. / g if g > 0 else 1.0
    (pre_hit_x, pre_hit_y), (pre_hit_dx, pre_hit_dy) = move_to_paddle_x(other_paddle_pos[0], ball_pos[0], ball_pos[1],
                                                                        ball_vel[0], ball_vel[1], direction,
                                                                        move_factor)
    goal_paddle_defend = TABLE_SIZE[1] / 2

    # find where the ball will hit
    lower_bound, upper_bound = calc_bounds(pre_hit_y)
    y1 = (upper_bound + lower_bound - 1) // 2 + 0.5
    y2 = y1 - 1

    hits = []  # hitting_paddle_y, ball_x, ball_y, ball_dx, ball_dy
    lowest_reach = other_paddle_pos[1] - t_to_hit * PADDLE_VEL
    highest_reach = other_paddle_pos[1] + t_to_hit * PADDLE_VEL
    while ((y1 <= upper_bound and y1 <= highest_reach) or (y2 >= lower_bound and y2 >= lowest_reach)) and calculating == -1:
        if y1 <= upper_bound and y1 <= highest_reach:
            hit = calc_hit(other_paddle_pos[0], y1, paddle_pos[0], pre_hit_x, pre_hit_y, pre_hit_dx, pre_hit_dy,
                           direction, move_factor)
            if hit is not None:
                hits.append(hit)
            y1 += 1
        if y2 >= lower_bound and y2 >= lowest_reach:
            hit = calc_hit(other_paddle_pos[0], y2, other_paddle_pos[0], pre_hit_x, pre_hit_y, pre_hit_dx, pre_hit_dy,
                           direction, move_factor)
            if hit is not None:
                hits.append(hit)
            y2 -= 1
    calculated = -1
    if debug: print("Done Predicting")


def towards():
    global goal_paddle_hit, calculated, hits, their_scores
    calculated = 0
    if debug: print("Optimizing my hit")

    # go to where the ball will hit
    g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
    move_factor = 1. / g if g > 0 else 1.0
    (pre_hit_x, pre_hit_y), (pre_hit_dx, pre_hit_dy) = move_to_paddle_x(paddle_pos[0], ball_pos[0], ball_pos[1],
                                                                        ball_vel[0], ball_vel[1], direction,
                                                                        move_factor)
    goal_paddle_hit = pre_hit_y
    calculated = 1
    if debug: print("Calculated basic hit")

    # find where the ball will hit
    lower_bound, upper_bound = calc_bounds(pre_hit_y)
    y1 = (upper_bound + lower_bound - 1) // 2 + 0.5
    y2 = y1 - 1

    hits = []  # hitting_paddle_y, ball_x, ball_y, ball_dx, ball_dy
    lowest_reach = paddle_pos[1] - t_to_hit * PADDLE_VEL
    highest_reach = paddle_pos[1] + t_to_hit * PADDLE_VEL
    while ((y1 <= upper_bound and y1 <= highest_reach) or (y2 >= lower_bound and y2 >= lowest_reach)) and calculating == 1:
        if y1 <= upper_bound and y1 <= highest_reach:
            hit = calc_hit(paddle_pos[0], y1, other_paddle_pos[0], pre_hit_x, pre_hit_y, pre_hit_dx, pre_hit_dy,
                           direction, move_factor)
            if hit is not None:
                hits.append(hit)
            y1 += 1
        if y2 >= lower_bound and y2 >= lowest_reach:
            hit = calc_hit(paddle_pos[0], y2, other_paddle_pos[0], pre_hit_x, pre_hit_y, pre_hit_dx, pre_hit_dy,
                           direction, move_factor)
            if hit is not None:
                hits.append(hit)
            y2 -= 1
    calculated = 2
    if debug: print("Calculated all hits")

    # use a thread to calculate one max move

    i = 0
    their_scores = {}
    while i < len(hits) and i < t_to_hit and calculating == 1 and False:
        hit = hits[i]
        y, hit_x, hit_y, hit_dx, hit_dy = hit

        if paddle_pos[1] - t_to_hit * PADDLE_VEL <= y <= paddle_pos[1] + t_to_hit * PADDLE_VEL:  # if we can reach it
            lower_bound, upper_bound = calc_bounds(pre_hit_y)
            y1 = lower_bound

            g = int((hit_dx ** 2 + hit_dy ** 2) ** .5)
            move_factor = 1. / g if g > 0 else 1.0
            best_score = -100000
            t_to_them = t_to_hit + (hit[1] - paddle_pos[0]) // hit[3]
            lowest_reach = other_paddle_pos[1] - t_to_them * PADDLE_VEL
            highest_reach = other_paddle_pos[1] + t_to_them * PADDLE_VEL
            while lower_bound <= y1 <= upper_bound and lowest_reach <= y1 <= highest_reach and calculating == 1:
                response = calc_hit(other_paddle_pos[0], y1, paddle_pos[0], hit_x, hit_y, hit_dx, hit_dy,
                                    -direction, move_factor)
                if response is not None:
                    t_to_react = (hit[1] - paddle_pos[0]) // hit[3] + (response[1] - other_paddle_pos[0]) // response[3]
                    t_to_intercept = (abs(response[2] - y) - PADDLE_SIZE[1] / 2) // PADDLE_VEL  #  i am conna by at y
                    score = t_to_intercept - t_to_react
                    if t_to_intercept > t_to_react:
                        score *= 10e5
                    else:
                        score *= hit[3]
                    if score > best_score:
                        best_score = score
                y1 += 1
            their_scores[hit[0]] = best_score
        i += 1

    calculated = 3
    if debug: print("Calculated returns", i)

def initialize(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET, ball_pos, ball_vel, paddle_pos, other_paddle_pos, direction
    global prev_ball_pos, prev_ball_vel, prev_paddle_pos, goal_paddle_y, prev_direction
    global goal_paddle_hit, goal_paddle_defend, t_since_hit, t_to_hit, calculating, calculated, hits, their_scores
    X_OFFSET = min(paddle_frect.pos[0], other_paddle_frect.pos[0]) + paddle_frect.size[0]
    TABLE_SIZE = (max(paddle_frect.pos[0], other_paddle_frect.pos[0]) - X_OFFSET - ball_frect.size[0],
                  table_size[1] - ball_frect.size[1])
    PADDLE_SIZE = (paddle_frect.size[0] + ball_frect.size[0], paddle_frect.size[1] + ball_frect.size[0])
    BALL_SIZE = (ball_frect.size[0], ball_frect.size[1])
    PADDLE_OFFSET = paddle_frect.size[1] / 2 - ball_frect.size[1] / 2
    ball_pos = [ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1]]
    ball_vel = (0, 0)
    paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
    other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)
    goal_paddle_hit = 0  # we want to move down at the start
    goal_paddle_defend = 0
    direction = 0
    t_since_hit = 0
    t_to_hit = 50
    calculating = 0
    calculated = 0
    hits = []
    their_scores = {}
    prev_ball_pos, prev_ball_vel, prev_paddle_pos, prev_other_paddle_pos, prev_direction = ball_pos, ball_vel, paddle_pos, other_paddle_pos, direction

time_per_t = 0.0001
time_per_search = 0.000023
inited = 0

debug = False
more_debug = False
timing = False

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global PADDLE_VEL, ball_pos, ball_vel, paddle_pos, other_paddle_pos, direction
    global prev_ball_pos, prev_ball_vel, prev_paddle_pos, prev_direction
    global t_since_hit, t_to_hit
    global scorer, towards_calculator, away_calculator, calculating, inited
    start = time.time_ns()
    # ball position is a list, so be careful
    # if first call
    #    do initialize
    if not inited:
        initialize(paddle_frect, other_paddle_frect, ball_frect, table_size)
        scorer = threading.Thread(target=process_hits, daemon=True)
        if debug: print("Initializing", X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET)
        inited = 1
    # if second call
    #   calc paddle vel
    elif inited == 1:
        PADDLE_VEL = abs(paddle_frect.pos[1] + PADDLE_OFFSET - prev_paddle_pos[1])
        if debug: print("2nd Call", PADDLE_VEL)
        scorer.start()
        inited = 2

    # calculate better positions
    new_ball_pos = (ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1])  # tuple so I don't change it
    new_paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
    new_other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)

    # calculate velocity
    new_ball_vel = ((new_ball_pos[0] - prev_ball_pos[0]), (new_ball_pos[1] - prev_ball_pos[1]))

    # figure out if we hit
    if new_paddle_pos[0] > new_other_paddle_pos[0]:
        side = 1
        new_paddle_pos = (new_paddle_pos[0] - BALL_SIZE[0], new_paddle_pos[1])
        new_other_paddle_pos = (new_other_paddle_pos[0] + PADDLE_SIZE[0] - BALL_SIZE[0], new_other_paddle_pos[1])
    else:
        side = -1
        new_paddle_pos = (new_paddle_pos[0] + PADDLE_SIZE[0] - BALL_SIZE[0], new_paddle_pos[1])
        new_other_paddle_pos = (new_other_paddle_pos[0] - BALL_SIZE[0], new_other_paddle_pos[1])
    new_direction = 1 if new_ball_vel[0] > 0 else -1
    if new_direction == side:
        hitting_paddle_pos = new_paddle_pos
        next_paddle_pos = new_other_paddle_pos
    else:
        hitting_paddle_pos = new_other_paddle_pos
        next_paddle_pos = new_paddle_pos

    # check if round reset
    if new_ball_pos[0] == TABLE_SIZE[0] / 2 and new_ball_pos[1] == TABLE_SIZE[1] / 2 and (prev_ball_pos[0] < 0 or TABLE_SIZE[0] < prev_ball_pos[0]):
        prev_direction = 0
        if debug: print("GAME OVER")

    if prev_direction != new_direction:
        t_since_hit = 0
        calculating = 0  # direction has changed so we have not done any calculations for this direction yet
    else:
        t_since_hit += 1

    # if velocity is valid
    if prev_ball_vel == new_ball_vel and t_since_hit > 1:  # if the velocities are reliable
        # update global values
        ball_pos = new_ball_pos
        ball_vel = new_ball_vel
        paddle_pos = new_paddle_pos
        other_paddle_pos = new_other_paddle_pos
        direction = new_direction
        t_to_hit = abs(-((new_paddle_pos[0] - ball_pos[0]) // -ball_vel[0]))

        if calculating == 0:  # if we have not started calculations
            if direction == side:
                towards_calculator = threading.Thread(target=towards, daemon=True)
                towards_calculator.start()
                if debug: print("Starting towards calculation")
                calculating = 1
            else:
                away_calculator = threading.Thread(target=away, daemon=True)
                away_calculator.start()
                if debug: print("Starting away calculation")
                calculating = -1

    # update previous values
    prev_ball_pos = new_ball_pos
    prev_ball_vel = new_ball_vel
    prev_paddle_pos = new_paddle_pos
    prev_direction = new_direction

    if direction == side:
        goal = goal_paddle_hit
    else:
        goal = goal_paddle_defend

    # move to goal_paddle_y
    if more_debug and goal == paddle_pos[1]: print("Arrived")
    elapsed = (time.time_ns() - start) / 10e6
    if elapsed > 0.1:
        print("pong_ai():", elapsed)

    if paddle_pos[1] > goal:
        return "up"
    elif paddle_pos[1] < goal:
        return "down"
    else:
        return "skip"

