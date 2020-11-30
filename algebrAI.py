####################
# Author: Ewan Wai #
####################

import math

initialized = False


def init(table_size):
    global initialized, prev_other_paddle_pos, prev_ball_pos

    prev_other_paddle_pos = (0, 0)
    prev_ball_pos = (table_size[0] / 2, table_size[1] / 2)
    initialized = True


def calc_vel(curr_pos, prev_pos):
    return (0.0001 if curr_pos[0] == prev_pos[0] else (curr_pos[0] - prev_pos[0]),
                curr_pos[1] - prev_pos[1])


def calc_intersect(ball_frect, ball_vel, paddle_frect, table_size):
    '''Calculates the height of the ball when it hits the paddle'''
    x = calc_paddle_bound(paddle_frect, ball_vel[0], ball_frect, table_size)
    y = ((ball_vel[1] * (x - ball_frect.pos[0]) / ball_vel[0]) + ball_frect.pos[1])

    return y


def scale_intersect(intersect, ball_frect, table_size):
    acc_table_height = table_size[1] - ball_frect.size[1]
    return ((intersect * (-1)**(intersect // acc_table_height)) % acc_table_height)


def calc_paddle_bound(paddle_frect, direction, ball_frect, table_size):
    return paddle_frect.pos[0] + (paddle_frect.size[0] if direction < 0 else -ball_frect.size[0])


def calc_hit(paddle_y, ball_frect, ball_vel, paddle_frect, other_paddle_frect, table_size):
    intersect = calc_intersect(ball_frect, ball_vel, paddle_frect, table_size)
    y = scale_intersect(intersect, ball_frect, table_size)

    center = paddle_frect.pos[1] + paddle_frect.size[1] / 2
    rel_dist_from_c = ((y + ball_frect.size[1] / 2 - paddle_y) / paddle_frect.size[1])
    rel_dist_from_c = min(0.5, rel_dist_from_c)
    rel_dist_from_c = max(-0.5, rel_dist_from_c)
    sign = 1 if ball_vel[0] > 0 else -1

    theta = sign * rel_dist_from_c * 45 * math.pi / 180

    v = [ball_vel[0], ball_vel[1] * (-1) ** (intersect // (table_size[1] - ball_frect.size[1]))]

    v = [math.cos(theta) * v[0] - math.sin(theta) * v[1],
         math.sin(theta) * v[0] + math.cos(theta) * v[1]]

    v[0] = -v[0]

    v = [math.cos(-theta) * v[0] - math.sin(-theta) * v[1],
         math.cos(-theta) * v[1] + math.sin(-theta) * v[0]]

    after_ball_frect = ball_frect.copy()
    after_ball_frect.pos = (calc_paddle_bound(paddle_frect, ball_vel[0], ball_frect, table_size), y)

    end_intersect = calc_intersect(after_ball_frect, v, other_paddle_frect, table_size)
    end_y = scale_intersect(end_intersect, after_ball_frect, table_size)

    return v[0], v[1], end_y

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global prev_other_paddle_pos, prev_ball_pos

    if not initialized:
        init(table_size)

    ball_vel = calc_vel(ball_frect.pos, prev_ball_pos)
    other_paddle_vel = calc_vel(other_paddle_frect.pos, prev_other_paddle_pos)

    direction = 2 * (paddle_frect.pos[0] > other_paddle_frect.pos[0]) - 1

    best_paddle_y = paddle_y = paddle_frect.pos[1] + paddle_frect.size[1] / 2
    other_paddle_y = other_paddle_frect.pos[1] + other_paddle_frect.size[1] / 2

    if direction * ball_vel[0] > 0: # moving towards me
        intersect = calc_intersect(ball_frect, ball_vel, paddle_frect, table_size)
        y = scale_intersect(intersect, ball_frect, table_size) + ball_frect.size[0] / 2

        best_d, best_score = 0, 0

        x = calc_paddle_bound(paddle_frect, direction, ball_frect, table_size)
        t = int((x - ball_frect.pos[0]) / ball_vel[0])
        scan = min(paddle_frect.size[1] // 2, t)
        for d in range(-scan, scan):
            after_vel_x, after_vel_y, end_y = calc_hit(y + d, ball_frect, ball_vel, paddle_frect, other_paddle_frect,
                                                       table_size)

            score = abs((end_y - other_paddle_y) / after_vel_x)
            if abs(end_y - other_paddle_y) - other_paddle_frect.size[1] / 2 > abs(
                    table_size[0] / after_vel_x):  # only make it go fast if we will win
                score = (end_y - other_paddle_y) * after_vel_x

            if score > best_score:
                best_d, best_score = d, score

        best_paddle_y = y + best_d

        trick_offset = math.copysign(1, best_paddle_y - best_paddle_y) * (x - ball_frect.pos[0]) / ball_vel[0]
        if abs(trick_offset) < 10:
            trick_offset = 0

        best_paddle_y = best_paddle_y + trick_offset

    else: # moving away from me
        intersect = calc_intersect(ball_frect, ball_vel, other_paddle_frect, table_size)
        y = scale_intersect(intersect, ball_frect, table_size) + ball_frect.size[0] / 2

        x = calc_paddle_bound(other_paddle_frect, -direction, ball_frect, table_size)
        t = abs(int((x - ball_frect.pos[0]) / ball_vel[0]))+1
        scan = min(other_paddle_frect.size[1] // 2, t)

        weighted_sum = 0
        sum_of_weights = 0
        best_y, best_score = -1, -1
        for d in range(-scan, scan):
            dx, dy, end_y = calc_hit(y + d, ball_frect, ball_vel, other_paddle_frect, paddle_frect, table_size)
            score = abs((end_y - paddle_y) * dx)
            weighted_sum += end_y * dx
            sum_of_weights += dx
            if score > best_score:
                best_y, best_score = end_y, score

        avg = weighted_sum / sum_of_weights

        best_paddle_y = (avg + best_y) / 2

    # best guess or middle if it goes to edges: 200 161, 200 192 vs donkey

    prev_ball_pos = ball_frect.pos
    prev_other_paddle_pos = other_paddle_frect.pos

    if paddle_y < best_paddle_y:
        return "down"
    elif paddle_y > best_paddle_y:
        return "up"
    else:
        return "none"
