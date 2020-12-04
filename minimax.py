# Ted Pinkerton and Ewan Wai

import math

max_angle = 45
math_pi = math.pi
paddle_bounce = 1.2
max_loop = 1000

from timeit import default_timer as timer


def move_to_paddle_x(paddle_x, ball_pos, ball_vel, table_size, direction, move_factor):
    """Calculates the position and velocity of the ball when it reaches the paddle_x
        Mutates the ball_pos
    """
    n = 0
    while (paddle_x - ball_pos[0]) * direction > 0:
        n += 1
        if(n>max_loop):
            break
        wall_y = -1
        if ball_vel[1] > 0:
            wall_y = table_size[1] + 1

        ticks_to_paddle = -int((paddle_x - ball_pos[0]) // (-ball_vel[0] * move_factor))
        ticks_to_wall = -int((wall_y - (ball_pos[1])) // (-ball_vel[1] * move_factor))  # ceiling computation

        if ticks_to_paddle < ticks_to_wall:
            ball_pos[0] += ball_vel[0] * ticks_to_paddle * move_factor
            ball_pos[1] += ball_vel[1] * ticks_to_paddle * move_factor
        else:
            ball_pos[0] += ball_vel[0] * move_factor * ticks_to_wall
            ball_pos[1] += ball_vel[1] * move_factor * ticks_to_wall
            # n += 1

            d = -((ball_pos[1] - wall_y) // (ball_vel[1] * -0.1 * move_factor))
            ball_pos[1] -= (2 * d) * ball_vel[1] * 0.1 * move_factor

            ball_vel = (ball_vel[0], -ball_vel[1])

    return ball_pos, ball_vel

def move_to_paddle(hitting_paddle_y, ball_pos, ball_vel, paddle_size, table_size, move_factor): # makes the ball actually hit the paddle
    if ball_pos[1] < hitting_paddle_y - paddle_size[1] / 2:  # if under paddle when at paddle level
        if ball_vel[1] > 0:
            ticks_to_paddle = -int((hitting_paddle_y - paddle_size[1] / 2 - ball_pos[1]) // (-ball_vel[1] * move_factor))
            ball_pos[0] += ball_vel[0] * move_factor * ticks_to_paddle
            ball_pos[1] += ball_vel[1] * move_factor * ticks_to_paddle
        else:  # ball will miss paddle
            return False
    elif ball_pos[1] > hitting_paddle_y + paddle_size[1] / 2:  # if above paddle when at paddle level
        if ball_vel[1] < 0:
            ticks_to_paddle = -int((hitting_paddle_y + paddle_size[1] / 2 - ball_pos[1]) // (-ball_vel[1] * move_factor))
            ball_pos[0] += ball_vel[0] * move_factor * ticks_to_paddle
            ball_pos[1] += ball_vel[1] * move_factor * ticks_to_paddle
        else:  # ball will miss paddle
            return False

    if ball_pos[0] <= -paddle_size[0] or ball_pos[0] >= table_size[0] + paddle_size[0]:  # if ball passed paddle
        return False

    return ball_pos


def do_hit(paddle_x, paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size, direction, move_factor, hit=False):
    c = 0

    while (ball_pos[1] > table_size[1] or ball_pos[1] < 0) or \
            (paddle_x - (paddle_size[0] if direction < 0 else 0) < ball_pos[0] < paddle_x + (
                    paddle_size[0] if direction > 0 else 0) and
             paddle_y - paddle_size[1] / 2 < ball_pos[1] < paddle_y + paddle_size[1] / 2):
        c += 1
        ball_pos[0] -= ball_vel[0] * 0.1 * move_factor
        ball_pos[1] -= ball_vel[1] * 0.1 * move_factor
        #print("adjustin", c)
        if (c > max_loop):
            break

    rel_dist_from_c = (ball_pos[1] - paddle_y) / (paddle_size[1] - ball_size[1])
    rel_dist_from_c = min(0.5, rel_dist_from_c)
    rel_dist_from_c = max(-0.5, rel_dist_from_c)

    theta = direction * rel_dist_from_c * 45 * math_pi / 180

    v = ball_vel

    v = [math.cos(theta) * v[0] - math.sin(theta) * v[1],
         math.sin(theta) * v[0] + math.cos(theta) * v[1]]

    v[0] = -v[0]

    v = [math.cos(-theta) * v[0] - math.sin(-theta) * v[1],
         math.cos(-theta) * v[1] + math.sin(-theta) * v[0]]

    # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
    if v[0] * -direction < 1:  # ball is not traveling (a) away from paddle (b) at a sufficient speed
        v[1] = (v[1] / abs(v[1])) * math.sqrt(
            v[0] ** 2 + v[1] ** 2 - 1)  # transform y velocity so as to maintain the speed
        v[
            0] = direction * -1  # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to increase by *1.2

    if not hit:
        ball_vel = (v[0] * paddle_bounce, v[1] * paddle_bounce)
    else:
        ball_vel = (v[0], v[1])

    while c > 0 or (paddle_x - (paddle_size[0] if direction < 0 else 0) < ball_pos[0] < paddle_x + (
            paddle_size[0] if direction > 0 else 0) and
                    paddle_y - paddle_size[1] / 2 < ball_pos[1] < paddle_y + paddle_size[
                        1] / 2):
        ball_pos[0] += ball_vel[0] * 0.1 * move_factor
        ball_pos[1] += ball_vel[1] * 0.1 * move_factor
        c -= 1

        if (c < -max_loop):
            break

    return ball_pos, ball_vel

def hit_paddle(paddle_x, paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size, direction):
    """Calculates the ball's position and velocity after hitting the paddle, assuming it is touching the paddle at the moment"""
    hit = False
    n = 0
    while not 0 < ball_pos[0] < table_size[0]:
        if n > max_loop:
            break
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g if g > 0 else 1.0
        if (paddle_x - (paddle_size[0] if direction < 0 else 0) < ball_pos[0] < paddle_x + (
                paddle_size[0] if direction > 0 else 0) and
                paddle_y - paddle_size[1] / 2 < ball_pos[1] < paddle_y + paddle_size[
                    1] / 2):
            #print("Hitting")
            ball_pos, ball_vel = do_hit(paddle_x, paddle_y, ball_pos, ball_vel, paddle_size, ball_size,
                                        table_size, direction, move_factor, hit)
            hit = True
        else:
            #print("Moving")
            ball_pos[0] += ball_vel[0] * move_factor
            ball_pos[1] += ball_vel[1] * move_factor

    return ball_pos, ball_vel


def calculate_ball_pos(left_paddle_y, right_paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size):
    """Calculate the y of the ball at the next paddle if ball has current pos and speed"""
    # collide

    # print "in wall. speed: ", self.speed
    direction = -1
    hitting_paddle_y = left_paddle_y
    x_bound = 0
    if ball_vel[0] > 0:
        direction = 1
        hitting_paddle_y = right_paddle_y
        x_bound = table_size[0]
    g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
    move_factor = 1. / g

    ball_pos, ball_vel = move_to_paddle_x(x_bound, ball_pos, ball_vel, table_size, direction, move_factor)


    ball_pos = move_to_paddle(hitting_paddle_y, ball_pos, ball_vel, paddle_size, table_size, move_factor)
    if not ball_pos:
        return False

    hit = False
    while not 0 < ball_pos[0] < table_size[0]:
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g
        if (x_bound - (paddle_size[0] if direction < 0 else 0) < ball_pos[0] < x_bound + (
            paddle_size[0] if direction > 0 else 0) and
                    hitting_paddle_y - paddle_size[1] / 2 < ball_pos[1] < hitting_paddle_y + paddle_size[
                        1] / 2):
            ball_pos, ball_vel = do_hit(x_bound, hitting_paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size, direction, move_factor, hit)
            hit = True
        else:
            ball_pos[0] += ball_vel[0] * move_factor
            ball_pos[1] += ball_vel[1] * move_factor

    direction = -1
    hitting_paddle_y = left_paddle_y
    x_bound = 0
    if ball_vel[0] > 0:
        direction = 1
        hitting_paddle_y = right_paddle_y
        x_bound = table_size[0]

    g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
    move_factor = 1. / g

    ball_pos, ball_vel = move_to_paddle_x(x_bound, ball_pos, ball_vel, table_size, direction, move_factor)

    #ball_pos = move_to_paddle(hitting_paddle_y, ball_pos, ball_vel, paddle_size, table_size, move_factor)
    #if not ball_pos:
     #   return False

    #print(ball_vel[0] * move_factor, ball_vel[1] * move_factor)

    return ball_pos, ball_vel



def old_move_to_paddle(left_paddle_y, right_paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size):
    direction = -1
    hitting_paddle_y = left_paddle_y
    x_bound = 0
    if ball_vel[0] > 0:
        direction = 1
        hitting_paddle_y = right_paddle_y
        x_bound = table_size[0]

    while (x_bound - ball_pos[0]) * direction > 0:
        # print(move_factor)
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g

        y_direction = -1
        wall_y = -1
        if ball_vel[1] > 0:
            y_direction = 1
            wall_y = table_size[1] + 1

        ticks_to_paddle = -int((x_bound - ball_pos[0]) // (-ball_vel[0] * move_factor))
        ticks_to_wall = -int((wall_y - (ball_pos[1])) // (-ball_vel[1] * move_factor))  # ceiling computation
        # print(ticks_to_paddle, ticks_to_wall)

        if ticks_to_paddle < ticks_to_wall:
            ball_pos[0] += ball_vel[0] * ticks_to_paddle * move_factor
            ball_pos[1] += ball_vel[1] * ticks_to_paddle * move_factor
        else:
            n = 0
            # while (int(ball_pos[1]) < wall_y and y_direction > 0) or (int(ball_pos[1]) > wall_y and y_direction < 0):
            ball_pos[0] += ball_vel[0] * move_factor * ticks_to_wall
            ball_pos[1] += ball_vel[1] * move_factor * ticks_to_wall
            # n += 1

            c = 0
            # print("bounce", [ball_pos[0], ball_pos[1]])
            while (y_direction < 0 and int(ball_pos[1]) < 0) or (y_direction > 0 and int(ball_pos[1]) > table_size[1]):
                c += 1
                ball_pos[0] -= ball_vel[0] * 0.1 * move_factor
                ball_pos[1] -= ball_vel[1] * 0.1 * move_factor

            ball_vel = (ball_vel[0], -ball_vel[1])
            while c > 0 or (y_direction < 0 and int(ball_pos[1]) < 0) or (
                    y_direction > 0 and int(ball_pos[1]) > table_size[1]):
                ball_pos[0] += ball_vel[0] * 0.1 * move_factor
                ball_pos[1] += ball_vel[1] * 0.1 * move_factor
                c -= 1
            # print("bounce", [ball_pos[0], ball_pos[1]])
            # ball_pos[0] += ball_vel[0] * (g) * move_factor
            # ball_pos[1] += ball_vel[1] * (g) * move_factor

    return ball_pos, ball_vel

def old_move_to_paddle2(left_paddle_y, right_paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size):
    direction = -1
    hitting_paddle_y = left_paddle_y
    x_bound = 0
    if ball_vel[0] > 0:
        direction = 1
        hitting_paddle_y = right_paddle_y
        x_bound = table_size[0]

    while (x_bound - ball_pos[0]) * direction > 0:
        # print(move_factor)
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g

        y_direction = -1
        wall_y = -1
        if ball_vel[1] > 0:
            y_direction = 1
            wall_y = table_size[1] + 1

        ticks_to_paddle = -int((x_bound - ball_pos[0]) // (-ball_vel[0] * move_factor))
        ticks_to_wall = -int((wall_y - (ball_pos[1])) // (-ball_vel[1] * move_factor))  # ceiling computation
        # print(ticks_to_paddle, ticks_to_wall)

        if ticks_to_paddle < ticks_to_wall:
            ball_pos[0] += ball_vel[0] * ticks_to_paddle * move_factor
            ball_pos[1] += ball_vel[1] * ticks_to_paddle * move_factor
        else:
            ball_pos[0] += ball_vel[0] * move_factor * ticks_to_wall
            ball_pos[1] += ball_vel[1] * move_factor * ticks_to_wall
            # n += 1

            d = -((ball_pos[1] - wall_y) // (ball_vel[1] * -0.1 * move_factor))
            ball_pos[1] -= (2 * d) * ball_vel[1] * 0.1 * move_factor

            ball_vel = (ball_vel[0], -ball_vel[1])

    return ball_pos, ball_vel

not_inited = 2
def init(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global not_inited, X_OFFSET, TABLE_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit, t_till_hit, round_end, hits
    X_OFFSET = min(paddle_frect.pos[0], other_paddle_frect.pos[0]) + paddle_frect.size[0]
    TABLE_SIZE = (max(paddle_frect.pos[0], other_paddle_frect.pos[0]) - X_OFFSET - ball_frect.size[0],
                  table_size[1] - ball_frect.size[1])
    PADDLE_SIZE = (paddle_frect.size[0] + ball_frect.size[0], paddle_frect.size[1] + ball_frect.size[0])
    PADDLE_OFFSET = paddle_frect.size[1] / 2 - ball_frect.size[1] / 2
    prev_ball_pos = (ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1])
    prev_paddle_pos = paddle_frect.pos
    goal_paddle_y = TABLE_SIZE[0]/2
    prev_direction = 0
    t_since_hit = 0
    t_till_hit = 0

"""
if start:
    save parameters
    move down to calculate
else:
    scale positions
    if round is over:
        go to middle
    
    if hit or spawn:
        calculate velocity
        calculate hit, time till
        
        if to me:
            start minmax
        if away:
            predict
    Move to position
"""

time_per_t = 0.0001
time_per_search = 0.000001
inited = 0
round_end = True
def initialize(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global inited, X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit, t_till_hit, round_end, hits
    X_OFFSET = min(paddle_frect.pos[0], other_paddle_frect.pos[0]) + paddle_frect.size[0]
    TABLE_SIZE = (max(paddle_frect.pos[0], other_paddle_frect.pos[0]) - X_OFFSET - ball_frect.size[0],
                  table_size[1] - ball_frect.size[1])
    PADDLE_SIZE = (paddle_frect.size[0] + ball_frect.size[0], paddle_frect.size[1] + ball_frect.size[0])
    BALL_SIZE = (ball_frect.size[0], ball_frect.size[1])
    PADDLE_OFFSET = paddle_frect.size[1] / 2 - ball_frect.size[1] / 2
    other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)
    prev_ball_pos = (ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1])
    inited = 1
    t_since_hit = 0
    t_till_hit = 50
    prev_direction = 0
    hits = []


def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global inited, X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_ball_vel, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit
    # if first call
    #    do initialize
    # if second call
    #   calc paddle vel
    #
    # calculate better positions
    # calculate velocity
    #
    # if velocity did change
    #
    # update previous values
    # move to goal_paddle_y

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global inited, X_OFFSET, TABLE_SIZE, BALL_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit, t_till_hit, round_end, hits
    if not inited:
        initialize(paddle_frect, other_paddle_frect, ball_frect, table_size)
        goal_paddle_y = 0  # make paddle go down so we can find paddle velocity
        inited = 1
        print("init")
    elif inited == 1:
        PADDLE_VEL = abs(paddle_frect.pos[1] + PADDLE_OFFSET - prev_paddle_pos[1])
        inited = 2

    ball_pos = [ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1]]
    paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
    other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)
    ball_vel = ((ball_pos[0] - prev_ball_pos[0]), (ball_pos[1] - prev_ball_pos[1]))
    prev_ball_pos = (ball_pos[0], ball_pos[1])

    if ball_pos[0] == prev_ball_pos[0] or ball_pos[1] == prev_ball_pos[1] and ball_vel[0] ** 2 + ball_vel[1] ** 2 >= 1:  # we can't get good data from a wall collision
        print("SCARY", t_since_hit, t_till_hit, ball_pos, prev_ball_pos)
        prev_ball_pos = (ball_pos[0], ball_pos[1])
        prev_paddle_pos = paddle_pos
        return "down"

    direction = -1
    hitting_paddle_x = 0
    next_paddle_x = TABLE_SIZE[0]
    if ball_vel[0] > 0:
        direction = 1
        hitting_paddle_x = TABLE_SIZE[0]
        next_paddle_x = 0
    else:
        direction = -1
        hitting_paddle_x = 0
        next_paddle_x = TABLE_SIZE[0]
    side = 1 if paddle_pos[0] > other_paddle_pos[0] else -1

    if ball_pos[0] < -PADDLE_SIZE[0] or ball_pos[0] > TABLE_SIZE[0] + PADDLE_SIZE[0]:  # round is ending
        goal_paddle_y = TABLE_SIZE[1] / 2
        prev_direction = 0
    else:
        t_since_hit = 0 if prev_direction != direction else t_since_hit + 1

        #if t_since_hit > 0:
        #print(t_since_hit)
        # find where ball will hit
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g if g > 0 else 1.0
        original_ball_pos = (ball_pos[0], ball_pos[1])
        ball_pos, ball_vel = move_to_paddle_x(hitting_paddle_x, ball_pos, ball_vel, TABLE_SIZE, direction, move_factor)
        ball_hit_pos = (ball_pos[0], ball_pos[1])
        goal_paddle_y = ball_pos[1]

        # find our search range and how long we get to search for
        t_till_hit = abs(-((ball_hit_pos[0] - original_ball_pos[0]) // -ball_vel[0]))

        if t_since_hit == 2:  # ball is is non new trajectory and we are confident it will continue
            hits = []
            if side == direction:  # if towards me
                hitting_paddle_y = paddle_pos[1]
                other_paddle_y = other_paddle_pos[1]
                goal_paddle_y = ball_hit_pos[1]

                #print(PADDLE_SIZE)

                upper_bound = min(int(ball_hit_pos[1] + PADDLE_SIZE[1] / 2) - 0.5,
                                  hitting_paddle_y + t_till_hit * PADDLE_VEL)
                upper_bound = min(upper_bound, TABLE_SIZE[1] - (PADDLE_SIZE[1] - BALL_SIZE[1]) / 2)
                lower_bound = max(int(ball_hit_pos[1] - PADDLE_SIZE[1] / 2) + 0.5,
                                  hitting_paddle_y - t_till_hit * PADDLE_VEL)
                lower_bound = max(lower_bound, (PADDLE_SIZE[1] - BALL_SIZE[1]) / 2)
                search_range = upper_bound - lower_bound
                t_to_search = t_till_hit - (
                        PADDLE_VEL * search_range / 2) - 2  # TODO: time we have until we need to book it to the edge

                n_searches = int(t_to_search * time_per_t / time_per_search)
                step = -(search_range // -n_searches)
                if step > 0:
                    goal_paddle_y = (upper_bound + lower_bound) / 2
                    hits = []
                    best_y, best_score = goal_paddle_y, -100000
                    for y in range(int(lower_bound), int(upper_bound)):
                        y = y + 0.5
                        this_ball_pos = [ball_hit_pos[0], ball_hit_pos[1]]
                        this_ball_vel = (ball_vel[0], ball_vel[1])
                        this_ball_pos = move_to_paddle(y, this_ball_pos, this_ball_vel, PADDLE_SIZE, TABLE_SIZE, move_factor)
                        if not this_ball_pos:
                            score = -1000000
                        else:
                            this_ball_pos, this_ball_vel = hit_paddle(hitting_paddle_x, y, this_ball_pos, this_ball_vel, PADDLE_SIZE, BALL_SIZE, TABLE_SIZE, direction)
                            g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
                            move_factor = 1. / g if g > 0 else 1.0
                            this_ball_pos, this_ball_vel = move_to_paddle_x(next_paddle_x, this_ball_pos, this_ball_vel, TABLE_SIZE, -direction, move_factor)
                            score = abs((this_ball_pos[1] - other_paddle_y) * this_ball_vel[0])
                            hits.append((y, (this_ball_pos[0], this_ball_pos[1]), (this_ball_vel[0], this_ball_vel[1])))
                        if score > best_score:
                            best_score = score
                            best_y = y
                    goal_paddle_y = best_y

                else:  # we may not make it in this case
                    goal_paddle_y = ball_hit_pos[1]
            else:
                hitting_paddle_y = other_paddle_pos[1]
                goal_paddle_y = TABLE_SIZE[1] / 2

        elif t_since_hit > 2:
            if side == direction:
                hitting_paddle_y = paddle_pos[1]
                other_paddle_y = other_paddle_pos[1]
                t_till_hit = abs(-((ball_hit_pos[0] - original_ball_pos[0]) // -ball_vel[0]))
                if len(hits) > 0:
                    best_y, best_score = -1, -100000
                    for hit in hits:
                        if abs(hit[0] - hitting_paddle_y) < t_till_hit * PADDLE_VEL:
                            score = abs((hit[1][1] - other_paddle_y))
                            if score > best_score:
                                best_y = hit[0]
                                best_score = score
                    if best_y != -1:
                        goal_paddle_y = best_y
                    else:
                        goal_paddle_y = ball_hit_pos[1]

    prev_direction = direction
    prev_paddle_pos = paddle_pos

    if paddle_pos[1] > goal_paddle_y:
        return "up"
    elif paddle_pos[1] < goal_paddle_y:
        return "down"
    else:
        return "skip"



def pong_ai2(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global not_inited, X_OFFSET, TABLE_SIZE, PADDLE_SIZE, PADDLE_OFFSET, PADDLE_VEL, prev_ball_pos, prev_paddle_pos, goal_paddle_y, prev_direction, t_since_hit, t_till_hit, round_end
    """
    paddles[0].frect.pos[1] - self.frect.size[1] / 2 + paddles[0].frect.size[1] / 2,
    paddles[1].frect.pos[1] - self.frect.size[1] / 2 + paddles[1].frect.size[1] / 2,
    [self.frect.pos[0] - 25, self.frect.pos[1]],
    self.speed,
    (paddles[0].frect.size[0] + self.frect.size[0], paddles[0].frect.size[1] + self.frect.size[1]),
    (15, 15), (375, 265))"""

    if not_inited:
        if not_inited == 2:
            not_inited = 1
            init(paddle_frect, other_paddle_frect, ball_frect, table_size)
            return "down"
        if not_inited == 1:
            PADDLE_VEL = abs(paddle_frect.pos[1] - prev_paddle_pos[1])
            not_inited = 0
            ball_pos = [ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1]]
            paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
            other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)
            ball_vel = (0.0000000001 if ball_pos[0] == prev_ball_pos[0] else (ball_pos[0] - prev_ball_pos[0]),
                        0.0000000001 if ball_pos[1] == prev_ball_pos[1] else (ball_pos[1] - prev_ball_pos[1]))  # prevent divisions by 0
            prev_ball_pos = (ball_pos[0], ball_pos[1])
            t_since_hit = 1
            return "up"

    ball_pos = [ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1]]
    paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
    other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)
    ball_vel = (0.0000000001 if ball_pos[0] == prev_ball_pos[0] else (ball_pos[0] - prev_ball_pos[0]),
                ball_pos[1] - prev_ball_pos[1])  # prevent divisions by 0
    prev_ball_pos = (ball_pos[0], ball_pos[1])

    if ball_pos[0] < -PADDLE_SIZE[0] or ball_pos[0] > TABLE_SIZE[0] + PADDLE_SIZE[0]: # if the ball is out of range:
            round_end = True
            goal_paddle_y = TABLE_SIZE[1] / 2
    elif round_end:
        not_inited = 1
    else:
        #print("prev_direction", prev_direction)
        ball_pos = [ball_frect.pos[0] - X_OFFSET, ball_frect.pos[1]]
        paddle_pos = (paddle_frect.pos[0] - X_OFFSET, paddle_frect.pos[1] + PADDLE_OFFSET)
        other_paddle_pos = (other_paddle_frect.pos[0] - X_OFFSET, other_paddle_frect.pos[1] + PADDLE_OFFSET)
        ball_vel = (0.0000000001 if ball_pos[0] == prev_ball_pos[0] else (ball_pos[0] - prev_ball_pos[0]),
                    ball_pos[1] - prev_ball_pos[1])  # prevent divisions by 0
        prev_ball_pos = (ball_pos[0], ball_pos[1])



        side = 1
        if paddle_pos[0] < other_paddle_pos[0]:
           side = -1

        direction = -1
        hitting_paddle_y = paddle_pos[1] if side == -1 else other_paddle_pos[1]
        hitting_paddle_x = 0
        next_paddle_x = TABLE_SIZE[0]
        if ball_vel[0] > 0:
            direction = 1
            hitting_paddle_y = paddle_pos[1] if side == 1 else other_paddle_pos[1]
            hitting_paddle_x, next_paddle_x = next_paddle_x, hitting_paddle_x

        t_since_hit = 0 if prev_direction != direction else t_since_hit + 1

        prev_direction = direction

        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g if g > 0 else 1

        if side == direction:
            if t_since_hit == 0:
                pass # TODO: KILL THREAD
        if t_since_hit == 2:
            # find where ball will hit
            ball_pos, ball_vel = move_to_paddle_x(hitting_paddle_x, ball_pos, ball_vel, TABLE_SIZE, direction, move_factor)
            ball_hit_pos = (ball_pos[0], ball_pos[1])
            goal_paddle_y = ball_pos[1]

            # find our search range and how long we get to search for
            t_till_hit = abs((TABLE_SIZE[0]-1) // ball_vel[0])
        if t_since_hit > t_till_hit:
            print(t_till_hit, t_since_hit)

    if paddle_pos[1] > goal_paddle_y:
        return "up"
    elif paddle_pos[1] < goal_paddle_y:
        return "down"
    else:
        return "skip"

if __name__ == "__main__":
    calculate_ball_pos(130, 130, [400,100], (1,1), (20,20), (15,15), (400, 280))
