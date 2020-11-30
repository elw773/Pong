# Ted Pinkerton and Ewan Wai

import math

max_angle = 45
math_pi = math.pi
paddle_bounce = 1.2

def move_to_paddle(left_paddle_y, right_paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size):
    pass

def calculate_ball_pos(left_paddle_y, right_paddle_y, ball_pos, ball_vel, paddle_size, ball_size, table_size, move_factor):
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

    while (x_bound-ball_pos[0]) * direction > 0:
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g
        #print(move_factor)

        y_direction = -1
        wall_y = -1
        if ball_vel[1] > 0:
            y_direction = 1
            wall_y = table_size[1]+1

        ticks_to_paddle = -int((x_bound - ball_pos[0]) // (-ball_vel[0] * move_factor))
        ticks_to_wall = -int((wall_y - (ball_pos[1])) // (-ball_vel[1] * move_factor)) # ceiling computation
        #print(ticks_to_paddle, ticks_to_wall)

        if ticks_to_paddle < ticks_to_wall:
            ball_pos[0] += ball_vel[0] * ticks_to_paddle * move_factor
            ball_pos[1] += ball_vel[1] * ticks_to_paddle * move_factor
        else:
            n = 0
            #while (int(ball_pos[1]) < wall_y and y_direction > 0) or (int(ball_pos[1]) > wall_y and y_direction < 0):
            ball_pos[0] += ball_vel[0] * move_factor * ticks_to_wall
            ball_pos[1] += ball_vel[1] * move_factor * ticks_to_wall
                #n += 1

            c = 0
            print("bounce", [ball_pos[0], ball_pos[1]])
            while (y_direction < 0 and int(ball_pos[1]) < 0) or (y_direction > 0 and int(ball_pos[1]) > table_size[1]):
                c += 1
                ball_pos[0] -= ball_vel[0] * 0.1 * move_factor
                ball_pos[1] -= ball_vel[1] * 0.1 * move_factor

            ball_vel = (ball_vel[0], -ball_vel[1])
            while c > 0 or (y_direction < 0 and int(ball_pos[1]) < 0) or (y_direction > 0 and int(ball_pos[1]) > table_size[1]):
                ball_pos[0] += ball_vel[0] * 0.1 * move_factor
                ball_pos[1] += ball_vel[1] * 0.1 * move_factor
                c -= 1
            print("bounce", [ball_pos[0], ball_pos[1]])
            #ball_pos[0] += ball_vel[0] * (g) * move_factor
            #ball_pos[1] += ball_vel[1] * (g) * move_factor


    hit = False
    while (ball_vel[0] > 0 and ball_pos[0] > 0) or (ball_vel[0] > 0 and ball_pos[0] > 0):
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1.0 / g

        paddle_height = paddle_size[1]
        x_size = ball_size[0] / 2 + paddle_size[0] / 2
        y_size = ball_size[1] / 2 + paddle_size[1] / 2
        if x_bound - x_size < ball_pos[0] < x_bound + x_size and hitting_paddle_y - y_size < ball_pos[1] < hitting_paddle_y + y_size:
            c = 0
            while (ball_vel[0] > 0 and ball_pos[0] >= table_size[0]) or (ball_vel[0] < 0 and ball_pos[0] <= 0) or\
                    (ball_vel[1] > 0 and ball_pos[1] >= table_size[1]) or (ball_vel[1] < 0 and ball_pos[1] <= 0) or\
                    ((ball_vel[1] > 0 and ball_pos[1] > hitting_paddle_y - paddle_height/2) or (ball_vel[1] < 0 and ball_pos[1] < hitting_paddle_y - paddle_height/2) and\
                     (ball_vel[0] > 0 and ball_pos[0] > table_size[0]) or (ball_vel[0] < 0 and ball_pos[0] < 0)):
                c += 1
                ball_pos[0] -= ball_vel[0] * 0.1 * move_factor
                ball_pos[1] -= ball_vel[1] * 0.1 * move_factor

            rel_dist_from_c = (ball_pos[1] - hitting_paddle_y) / paddle_height
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
            if v[0] * direction < 1:  # ball is not traveling (a) away from paddle (b) at a sufficient speed
                print("sketch")
                v[1] = (v[1] / abs(v[1])) * math.sqrt(v[0] ** 2 + v[1] ** 2 - 1)  # transform y velocity so as to maintain the speed
                v[0] = direction * -1 # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to increase by *1.2

            if not hit:
                ball_vel = (v[0] * paddle_bounce, v[1] * paddle_bounce)
            else:
                ball_vel = (v[0], v[1])
            hit = True

            while c > 0 or ((ball_vel[1] < 0 and ball_pos[1] > hitting_paddle_y - paddle_height/2) or (ball_vel[1] > 0 and ball_pos[1] < hitting_paddle_y - paddle_height/2) and\
                     (ball_vel[0] < 0 and ball_pos[0] > table_size[0]) or (ball_vel[0] > 0 and ball_pos[0] < 0)):
                ball_pos[0] += ball_vel[0] * 0.1 * move_factor
                ball_pos[1] += ball_vel[1] * 0.1 * move_factor
                c -= 1
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

    while (x_bound - ball_pos[0]) * direction > 0:
        g = int((ball_vel[0] ** 2 + ball_vel[1] ** 2) ** .5)
        move_factor = 1. / g
        # print(move_factor)

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
            print("bounce", [ball_pos[0], ball_pos[1]])
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
            print("bounce", [ball_pos[0], ball_pos[1]])
            # ball_pos[0] += ball_vel[0] * (g) * move_factor
            # ball_pos[1] += ball_vel[1] * (g) * move_factor

    return ball_pos