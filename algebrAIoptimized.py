####################
# Author: Ewan Wai #
####################

import math
from timeit import default_timer as timer
import time

initialized = False
paddle_bounce = 1.2


def init(table_size):
    global initialized, prev_other_paddle_pos, prev_ball_pos, prev_ball_vel, prev_best_paddle_y, data

    prev_other_paddle_pos = (0, 0)
    prev_ball_pos = (table_size[0] / 2, table_size[1] / 2)
    prev_best_paddle_y = table_size[1] / 2
    prev_ball_vel =(0, 0)
    data = {"i":0, "scan":0, "best_d":0, "best_score":0, "best_y":0, "weighted_sum":0, "sum_of_weights":0}
    initialized = True


def calc_vel(curr_pos, prev_pos):
    return (0.0001 if curr_pos[0] == prev_pos[0] else (curr_pos[0] - prev_pos[0]),
                curr_pos[1] - prev_pos[1])


def calc_intersect(ball_pos, ball_size, ball_vel, paddle_pos, paddle_size):
    '''Calculates the height of the ball when it hits the paddle'''
    x = calc_paddle_bound(paddle_pos, paddle_size, ball_vel[0], ball_size)
    y = ((ball_vel[1] * (x - ball_pos[0]) / ball_vel[0]) + ball_pos[1])

    return y


def scale_intersect(intersect, ball_size, table_size):
    acc_table_height = table_size[1] - ball_size[1]
    return (intersect * (-1) ** (intersect // acc_table_height)) % acc_table_height


def calc_paddle_bound(paddle_pos, paddle_size, direction, ball_size):
    return paddle_pos[0] + (paddle_size[0] if direction < 0 else -ball_size[0])


def calc_hit(paddle_y, ball_pos, ball_size, ball_vel, paddle_pos, paddle_size, other_paddle_pos, other_paddle_size, table_size):
    intersect = calc_intersect(ball_pos, ball_size, ball_vel, paddle_pos, paddle_size)
    y = scale_intersect(intersect, ball_size, table_size)

    center = paddle_pos[1] + paddle_size[1] / 2
    rel_dist_from_c = ((y + ball_size[1] / 2 - int(paddle_y)) / paddle_size[1])
    rel_dist_from_c = min(0.5, rel_dist_from_c)
    rel_dist_from_c = max(-0.5, rel_dist_from_c)
    sign = 1 if ball_vel[0] > 0 else -1

    theta = sign * rel_dist_from_c * 45 * math.pi / 180

    v = [ball_vel[0], ball_vel[1] * (-1) ** (intersect // (table_size[1] - ball_size[1]))]

    v = [math.cos(theta) * v[0] - math.sin(theta) * v[1],
         math.sin(theta) * v[0] + math.cos(theta) * v[1]]

    v[0] = -v[0]

    v = [math.cos(-theta) * v[0] - math.sin(-theta) * v[1],
         math.cos(-theta) * v[1] + math.sin(-theta) * v[0]]
    v = (v[0] * paddle_bounce, v[1] * paddle_bounce)

    after_ball_pos = (calc_paddle_bound(paddle_pos, paddle_size, ball_vel[0], ball_size), y)

    end_intersect = calc_intersect(after_ball_pos, ball_size, v, other_paddle_pos, other_paddle_size)
    end_y = scale_intersect(end_intersect, ball_size, table_size)

    return v[0], v[1], end_y

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global prev_other_paddle_pos, prev_ball_pos, prev_ball_vel, prev_best_paddle_y, data
    start = timer()
    compute_loops = 4

    if not initialized:
        init(table_size)

    paddle_pos, paddle_size = paddle_frect.pos, paddle_frect.size
    other_paddle_pos, other_paddle_size = other_paddle_frect.pos, other_paddle_frect.size
    ball_pos, ball_size = ball_frect.pos, ball_frect.size

    ball_vel = calc_vel(ball_pos, prev_ball_pos)
    other_paddle_vel = calc_vel(other_paddle_pos, prev_other_paddle_pos)

    direction = 2 * (paddle_pos[0] > other_paddle_pos[0]) - 1

    paddle_y = paddle_pos[1] + paddle_size[1] / 2
    other_paddle_y = other_paddle_pos[1] + other_paddle_size[1] / 2

    if prev_ball_vel[0] * ball_vel[0] < 0:  # if ball bounced off paddle
        data["i"] = 0  # restart calculations

    if direction * ball_vel[0] > 0:  # moving towards me
        intersect = calc_intersect(ball_pos, ball_size, ball_vel, paddle_pos, paddle_size)
        y = scale_intersect(intersect, ball_size, table_size) + ball_size[0] / 2

        x = calc_paddle_bound(paddle_pos, paddle_size, direction, ball_size)
        t = int((x - ball_pos[0]) / ball_vel[0]) - compute_loops

        if data["i"] == 0:
            for key in data:
                data[key] = 0
            data["scan"] = min(paddle_size[1] // 2, t)

        scan = 2.0*data["scan"]/compute_loops
        low_bound = -data["scan"] + (scan*data["i"])

        for d in range(int(low_bound), int(low_bound+scan)):
            after_vel_x, after_vel_y, end_y = calc_hit(y + d, ball_pos, ball_size, ball_vel, paddle_pos, paddle_size,
                                                       other_paddle_pos, other_paddle_size, table_size)

            t = int(table_size[0]  / ball_vel[0])
            scan2 = min(other_paddle_frect.size[1] // 2, t)

            if abs(end_y - other_paddle_y) - other_paddle_size[1] / 2 > abs(
                    table_size[0] / after_vel_x):  # only make it go fast if we will win
                score = (end_y - other_paddle_y) * after_vel_x * 100
            else:
                max_their_score = 1
                for d2 in range(-scan2, scan2):
                    after_vel_x2, after_vel_y2, end_y2 = calc_hit(end_y + d2, (x, y), ball_size, (after_vel_x, after_vel_y), other_paddle_pos,
                                                               other_paddle_size,
                                                               paddle_pos, paddle_size, table_size)

                    score = abs((end_y2 - (y+d))) * after_vel_x2
                    if abs((end_y2 - (y+d))) - paddle_size[1] / 2 > abs(
                            table_size[0] / after_vel_x2):  # only make it go fast if we will win
                        score = (end_y2 - (end_y2 - (y+d))) * after_vel_x2 * 100
                    max_their_score = max(score, max_their_score)
                score = 1 / max_their_score


            if score > data["best_score"]:
                data["best_d"], data["best_score"] = d, score

        data["i"] += 1

        if data["i"] > compute_loops:
            data["i"] = 0
            best_paddle_y = y + data["best_d"]
            prev_best_paddle_y = best_paddle_y
        else:
            best_paddle_y = prev_best_paddle_y

        trick_offset = math.copysign(1, best_paddle_y - best_paddle_y) * (x - ball_pos[0]) / ball_vel[0]
        if abs(trick_offset) < 10:
            trick_offset = 0

        best_paddle_y = best_paddle_y #+ trick_offset

    else:  # moving away from me
        intersect = calc_intersect(ball_pos, ball_size, ball_vel, other_paddle_pos, other_paddle_size)
        y = scale_intersect(intersect, ball_size, table_size) + ball_size[0] / 2

        x = calc_paddle_bound(other_paddle_pos, other_paddle_size, -direction, ball_size)
        t = abs(int((x - ball_pos[0]) / ball_vel[0])) - compute_loops
        if data["i"] == 0:
            for key in data:
                data[key] = 0
            data["scan"] = min(other_paddle_size[1] // 2, t)

        scan = 2.0 * data["scan"] / compute_loops
        low_bound = -data["scan"] + (scan * data["i"])

        for d in range(int(low_bound), int(low_bound + scan)):
            dx, dy, end_y = calc_hit(y + d, ball_pos, ball_size, ball_vel, other_paddle_pos, other_paddle_size,
                                     paddle_pos, paddle_size, table_size)
            score = abs((end_y - paddle_y) * dx)
            data["weighted_sum"] += end_y * dx
            data["sum_of_weights"] += dx
            if score > data["best_score"]:
                data["best_y"], data["best_score"] = end_y, score

        data["i"] += 1

        if data["i"] > compute_loops:
            data["i"] = 0

            best_paddle_y = data["best_y"]
            if data["sum_of_weights"] != 0:
                avg = data["weighted_sum"] / data["sum_of_weights"]
                best_paddle_y = (avg + data["best_y"]) / 2
            prev_best_paddle_y = best_paddle_y
        else:
            best_paddle_y = prev_best_paddle_y

    # best guess or middle if it goes to edges: 200 161, 200 192 vs donkey

    prev_ball_pos = ball_pos
    prev_other_paddle_pos = other_paddle_pos
    prev_ball_vel = ball_vel

    end = timer()
    if (end - start) * 1000 > 0.1:
        pass#print((end - start) * 1000)

    if paddle_y < best_paddle_y:
        return "down"
    elif paddle_y > best_paddle_y:
        return "up"
    else:
        return "none"