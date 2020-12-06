# Ted Pinkerton and Ewan Wai

import math
import numpy as np

max_angle = 45
math_pi = math.pi
paddle_bounce = 1.2
max_loop = 400

import threading
from timeit import default_timer as timer
import time
import numpy


def sigmoid(x):
    y = numpy.exp(x)
    return y / (1 + y)


def move_to_paddle_x(paddle_x, ball_x, ball_y, ball_dx, ball_dy, table_size, direction, move_factor):
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
            wall_y = table_size[1] + 1

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
             pre_hit_ball_dy, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, move_factor):
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
                                                                TABLE_SIZE, -direction, move_factor)

        return hitting_paddle_y, ball_x, ball_y, ball_dx, ball_dy


def calc_hits(lower_bound, upper_bound, step, hitting_paddle_pos, next_paddle_pos, t_till_hit, pre_hit_ball_pos,
              pre_hit_ball_vel, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, flag, done):
    global calculating, results
    y1 = lower_bound
    y2 = upper_bound
    calculating = flag
    results = []  # [paddle_y, hit_x, hit_y, hit_velx, hit_vely]

    g = int((pre_hit_ball_vel[0] ** 2 + pre_hit_ball_vel[1] ** 2) ** .5)
    pre_hit_move_factor = 1. / g if g > 0 else 1.0

    while calculating == flag and y1 < y2:
        result = calc_hit(hitting_paddle_pos[0], y1, next_paddle_pos[0], pre_hit_ball_pos[0], pre_hit_ball_pos[1],
                          pre_hit_ball_vel[0], pre_hit_ball_vel[1], TABLE_SIZE, BALL_SIZE,
                          PADDLE_SIZE, PADDLE_VEL, direction, pre_hit_move_factor)
        if not result is None:
            results.append(result)

        result = calc_hit(hitting_paddle_pos[0], y2, next_paddle_pos[0], pre_hit_ball_pos[0], pre_hit_ball_pos[1],
                          pre_hit_ball_vel[0], pre_hit_ball_vel[1], TABLE_SIZE, BALL_SIZE,
                          PADDLE_SIZE, PADDLE_VEL, direction, pre_hit_move_factor)
        if not result is None:
            results.append(result)

        y1 += step
        y2 -= step

    if y1 == y2:
        result = calc_hit(hitting_paddle_pos[0], y1, next_paddle_pos[0], pre_hit_ball_pos[0], pre_hit_ball_pos[1],
                          pre_hit_ball_vel[0], pre_hit_ball_vel[1], TABLE_SIZE, BALL_SIZE,
                          PADDLE_SIZE, PADDLE_VEL, direction, pre_hit_move_factor)
        if not result is None:
            results.append(result)

    calculating = done
    if debug: print("Done calculations", done)


def calc_posibilities(hitting_paddle_pos, next_paddle_pos, t_till_hit, pre_hit_ball_pos, pre_hit_ball_vel, TABLE_SIZE,
                      BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, flag, done):
    global calculating, calc_thread
    start = timer()
    highest_reach = hitting_paddle_pos[1] + (t_till_hit) * PADDLE_VEL
    upper_end = int(pre_hit_ball_pos[1] + PADDLE_SIZE[1] / 2) + 10.5
    upper_limit = TABLE_SIZE[1] - PADDLE_SIZE[1] / 2 + BALL_SIZE[1]

    lowest_reach = hitting_paddle_pos[1] - (t_till_hit) * PADDLE_VEL
    lower_end = int(pre_hit_ball_pos[1] - PADDLE_SIZE[1] / 2) - 10.5
    lower_limit = PADDLE_SIZE[1] / 2 - BALL_SIZE[1]

    upper_bound = min(highest_reach, upper_end, upper_limit)
    lower_bound = max(lowest_reach, lower_end, lower_limit)

    search_range = upper_bound - lower_bound

    t_to_search = t_till_hit - (PADDLE_VEL * search_range / 2) - 2  # safety net of 2 ticks

    n_searches = int(t_to_search * time_per_t / time_per_search)

    if search_range < 0 or n_searches == 0:
        calculating = 3
        return

    step = -int(search_range // -n_searches)

    if debug: print("Searching {} to {} ({}), doing {} by {} in {}".format(lower_bound, upper_bound,
                                                                           upper_bound - lower_bound,
                                                                           n_searches,
                                                                           step,
                                                                           t_to_search))

    results = []

    calc_thread = threading.Thread(target=calc_hits, args=(
    lower_bound, upper_bound, step, hitting_paddle_pos, next_paddle_pos, t_till_hit,
    pre_hit_ball_pos, pre_hit_ball_vel, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, flag, done),
                                   daemon=True)
    calc_thread.start()

    # if timing: print("Posibilities", (timer() - start), search_range * time_per_search / time_per_t, t_to_search)

    # calc_hits(lower_bound, upper_bound, step, hitting_paddle_pos, next_paddle_pos, t_till_hit,
    #         pre_hit_ball_pos, pre_hit_ball_vel, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, flag, done)


time_per_t = 0.0001
time_per_search = 0.000023
inited = 0
round_end = True


def initialize(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global inited, X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_ball_vel, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit, calculating, done_calculating, results, calc_thread
    X_OFFSET = min(paddle_frect.pos[0], other_paddle_frect.pos[0]) + paddle_frect.size[0]
    TABLE_SIZE = (max(paddle_frect.pos[0], other_paddle_frect.pos[0]) - X_OFFSET - ball_frect.size[0],
                  table_size[1] - ball_frect.size[1])
    PADDLE_SIZE = (paddle_frect.size[0] + ball_frect.size[0], paddle_frect.size[1] + ball_frect.size[0])
    BALL_SIZE = (ball_frect.size[0], ball_frect.size[1])
    PADDLE_OFFSET = paddle_frect.size[1] / 2 - ball_frect.size[1] / 2
    prev_ball_pos = [ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1]]
    prev_ball_vel = (0, 0)
    prev_paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
    goal_paddle_y = 0  # we want to move down at the start
    prev_direction = 0
    t_since_hit = 0
    t_till_hit = 50
    calculating = 0
    done_calculating = True
    results = []


debug = False
more_debug = False
timing = True


def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global inited, X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_ball_vel, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit, t_till_hit, calculating, results
    start = time.time_ns()
    # ball position is a list, so be careful
    # if first call
    #    do initialize
    if not inited:
        initialize(paddle_frect, other_paddle_frect, ball_frect, table_size)
        if debug: print("Initializing", X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET)
        inited = 1
    # if second call
    #   calc paddle vel
    elif inited == 1:
        PADDLE_VEL = abs(paddle_frect.pos[1] + PADDLE_OFFSET - prev_paddle_pos[1])
        if debug: print("2nd Call", PADDLE_VEL)
        inited = 2

    # calculate better positions
    ball_pos = (ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1])  # tuple so I don't change it
    paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
    other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)

    # calculate velocity
    ball_vel = ((ball_pos[0] - prev_ball_pos[0]), (ball_pos[1] - prev_ball_pos[1]))

    # figure our if we hit
    if paddle_pos[0] > other_paddle_pos[0]:
        side = 1
        paddle_pos = (paddle_pos[0] - BALL_SIZE[0], paddle_pos[1])
        other_paddle_pos = (other_paddle_pos[0] + PADDLE_SIZE[0] - BALL_SIZE[0], other_paddle_pos[1])
    else:
        side = -1
        paddle_pos = (paddle_pos[0] + PADDLE_SIZE[0] - BALL_SIZE[0], paddle_pos[1])
        other_paddle_pos = (other_paddle_pos[0] - BALL_SIZE[0], other_paddle_pos[1])
    direction = 1 if ball_vel[0] > 0 else -1

    if direction == side:
        hitting_paddle_pos = paddle_pos
        next_paddle_pos = other_paddle_pos
    else:
        hitting_paddle_pos = other_paddle_pos
        next_paddle_pos = paddle_pos

    # check if round reset
    if ball_pos[0] == TABLE_SIZE[0] / 2 and ball_pos[1] == TABLE_SIZE[1] / 2 and (
            prev_ball_pos[0] < 0 or TABLE_SIZE[0] < prev_ball_pos[0]):
        goal_paddle_y = TABLE_SIZE[1] / 2
        prev_direction = 0
        if debug: print("GAME OVER")

    if prev_direction != direction:
        t_since_hit = 0
        calculating = 0  # direction has changed so we have not done any calculations for this direction yet
    else:
        t_since_hit += 1

    # if velocity is valid
    if prev_ball_vel == ball_vel and t_since_hit > 1:  # if the velocities are reliable
        #   calculate the first intercept
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g if g > 0 else 1.0

        pre_hit_ball_pos, pre_hit_ball_vel = move_to_paddle_x(hitting_paddle_pos[0], ball_pos[0], ball_pos[1],
                                                              ball_vel[0], ball_vel[1], TABLE_SIZE, direction,
                                                              move_factor)

        t_till_hit = abs(-((pre_hit_ball_pos[0] - ball_pos[0]) // -ball_vel[0]))

        if direction == side:  # if ball is coming
            if calculating < 1:  # | -2 done predict | -1 doing predict | 0 none | 1 doing minmax | 2 done minmax | 3 minimax not viable
                if debug: print("Starting Minimax")
                calc_posibilities(hitting_paddle_pos, next_paddle_pos, t_till_hit, pre_hit_ball_pos,
                                  pre_hit_ball_vel, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, 1, 2)
            goal_paddle_y = pre_hit_ball_pos[1]
        else:
            if calculating > -1:
                if debug: print("Starting Prediction")
                calc_posibilities(hitting_paddle_pos, next_paddle_pos, t_till_hit, pre_hit_ball_pos,
                                  pre_hit_ball_vel, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_VEL, direction, -1, -2)
            goal_paddle_y = TABLE_SIZE[1] / 2
        elapsed = (time.time_ns() - start) / 1e6
    if ball_vel[0] != 0 and ball_vel[1] != 0:
        if direction == side and calculating == 2:
            best_y, best_score = -1, -1
            highest_reach = hitting_paddle_pos[1] + t_till_hit * PADDLE_VEL
            lowest_reach = hitting_paddle_pos[1] - t_till_hit * PADDLE_VEL
            for hit in results:
                if lowest_reach < hit[0] < highest_reach:
                    t_to_them = t_till_hit + (hit[1] - paddle_pos[0]) // hit[3] + 30
                    t_to_intercept = (abs(hit[2] - other_paddle_pos[1]) - PADDLE_SIZE[1] / 2) // PADDLE_VEL
                    if t_to_intercept > t_to_them:
                        score = t_to_intercept - t_to_them
                        if score > best_score:
                            best_y, best_score = hit[0], score
            if best_y > -1:
                goal_paddle_y = best_y
                if debug: print("Predict I win", best_score)
        if direction * side == -1 and calculating == -2:
            g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
            move_factor = 1. / g if g > 0 else 1.0

            pre_hit_ball_pos, pre_hit_ball_vel = move_to_paddle_x(hitting_paddle_pos[0], ball_pos[0], ball_pos[1],
                                                                  ball_vel[0], ball_vel[1], TABLE_SIZE, direction,
                                                                  move_factor)

            prediction = calc_hit(hitting_paddle_pos[0], hitting_paddle_pos[1], next_paddle_pos[0],
                                  pre_hit_ball_pos[0], pre_hit_ball_pos[1], pre_hit_ball_vel[0],
                                  pre_hit_ball_vel[1], TABLE_SIZE, BALL_SIZE, PADDLE_SIZE,
                                  PADDLE_VEL, direction, move_factor)

            sum, sum_of_weights = 0, 0
            highest_reach = hitting_paddle_pos[1] + t_till_hit * PADDLE_VEL
            lowest_reach = hitting_paddle_pos[1] - t_till_hit * PADDLE_VEL
            for hit in results:
                if lowest_reach < hit[0] < highest_reach:
                    t_to_me = t_till_hit + (hit[1] - other_paddle_pos[0]) // hit[3]
                    t_to_intercept = (abs(hit[2] - paddle_pos[1]) - PADDLE_SIZE[1] / 2) // PADDLE_VEL
                    weight = hit[3]
                    if weight <= 0:
                        weight = 0
                    sum += hit[0] * weight
                    sum_of_weights += weight
            if sum_of_weights != 0 and t_till_hit * PADDLE_VEL > PADDLE_SIZE[1] / 2:
                goal_paddle_y = ((sum / sum_of_weights) + TABLE_SIZE[1]) / 2
            if not prediction is None:
                # if t_till_hit * PADDLE_VEL > PADDLE_SIZE[1]/2:
                #    goal_paddle_y = (prediction[2] + TABLE_SIZE[1]) / 2
                # else:
                goal_paddle_y = prediction[2]
                if more_debug: print("Predict", pre_hit_ball_pos[1], goal_paddle_y)

    # update previous values
    prev_ball_pos = ball_pos
    prev_ball_vel = ball_vel
    prev_paddle_pos = paddle_pos
    prev_direction = direction

    # move to goal_paddle_y
    if more_debug and goal_paddle_y == paddle_pos[1]: print("Arrived")
    elapsed = (time.time_ns() - start / 10e6)
    if timing and elapsed > 0.1:
        print("pong_ai():", elapsed)
    if paddle_pos[1] > goal_paddle_y:
        return "up"
    elif paddle_pos[1] < goal_paddle_y:
        return "down"
    else:
        return "skip"
