import math

# ------------------ #
# my code (POGGERS) #
# ----------------- #
# this one is just better

def init(table_size):
    global prev_ball_pos, ball_speed, prev_ball_speed
    prev_ball_pos = (table_size[0]/2, table_size[1]/2)
    prev_ball_speed = (0.1, 0.1)
    ball_speed = (0.1, 0.1)


def get_ball_speed(prev_pos, curr_pos):
    if curr_pos[0]-prev_pos[0] == 0:
        return (0.0000001, curr_pos[1] - prev_pos[1])
    return (curr_pos[0]-prev_pos[0], curr_pos[1]-prev_pos[1])

#
# 20+paddle_size
#   |
#   |
#   |
#
#
#
#

def get_hit_height(x, y, dx, dy, left_bound, right_bound, height):
    if dx > 0:
        hit_height = ((dy * (left_bound - x) / dx) + y)
    else:
        hit_height = ((dy * (right_bound - x) / dx) + y)

    return (hit_height * (-1)**(hit_height // height)) % height

first = True

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    '''return "up" or "down", depending on which way the paddle should go to
    align its centre with the centre of the ball, assuming the ball will
    not be moving

    Arguments:
    paddle_frect: a rectangle representing the coordinates of the paddle
                  paddle_frect.pos[0], paddle_frect.pos[1] is the top-left
                  corner of the rectangle.
                  paddle_frect.size[0], paddle_frect.size[1] are the dimensions
                  of the paddle along the x and y axis, respectively

    other_paddle_frect:
                  a rectangle representing the opponent paddle. It is formatted
                  in the same way as paddle_frect
    ball_frect:   a rectangle representing the ball. It is formatted in the
                  same way as paddle_frect
    table_size:   table_size[0], table_size[1] are the dimensions of the table,
                  along the x and the y axis respectively

    The coordinates look as follows:

     0             x
     |------------->
     |
     |
     |
 y   v
    '''
    global first, prev_ball_pos, ball_speed, prev_ball_speed
    if first:
        init(table_size)
        first = False

    ball_speed = get_ball_speed(ball_frect.pos, prev_ball_pos)

    direction = 1

    left_paddle_bound = other_paddle_frect.pos[0] + other_paddle_frect.size[0]
    right_paddle_bound = paddle_frect.pos[0] - ball_frect.size[0]

    if paddle_frect.pos[0] < other_paddle_frect.pos[0]:
        right_paddle_bound, left_paddle_bound = left_paddle_bound, right_paddle_bound
        direction = -1

    actual_table_height = table_size[1] - ball_frect.size[1]

    ball_pos = ball_frect.pos

    y = get_hit_height(ball_pos[0], ball_pos[1], ball_speed[0], ball_speed[1], left_paddle_bound, right_paddle_bound, actual_table_height)
    if ball_speed[0] > 0:
        hit_height = ((ball_speed[1] * (left_paddle_bound - ball_pos[0]) / ball_speed[0]) + y)
    else:
        hit_height = ((ball_speed[1] * (right_paddle_bound - ball_pos[0]) / ball_speed[0]) + y)

    #print("\t" + str(y))
    if prev_ball_speed[0] * ball_speed[0] < 0:
        pass#print(ball_speed)

    opponent_y = y

    v = [ball_speed[0], ball_speed[1] * (-1)**(hit_height // actual_table_height) ]

    if direction * ball_speed[0] > 0: # going away
        distance = right_paddle_bound - ball_pos[0]
        if direction > 0:
            distance = left_paddle_bound - ball_pos[0]
        time = distance / ball_speed[0]
        opponent_paddle_y = y
        if abs(other_paddle_frect.pos[1] - y) > time:
            opponent_paddle_y = other_paddle_frect.pos[1] + (math.copysign(1,y - other_paddle_frect.pos[1]) * time)
        #opponent_paddle_y = other_paddle_frect.pos[1]


        x = left_paddle_bound if ball_speed[0] > 0 else right_paddle_bound

        center = opponent_paddle_y + other_paddle_frect.size[1] / 2
        rel_dist_from_c = ((ball_pos[1]+.5*ball_frect.size[1] - center) / other_paddle_frect.size[1])
        rel_dist_from_c = min(0.5, rel_dist_from_c)
        rel_dist_from_c = max(-0.5, rel_dist_from_c)
        sign = 1 * direction

        theta = sign * rel_dist_from_c * 45 * math.pi / 180

        v = [ball_speed[0], ball_speed[1] * (-1)**(hit_height // actual_table_height) ]

        v = [math.cos(theta) * v[0] - math.sin(theta) * v[1],
             math.sin(theta) * v[0] + math.cos(theta) * v[1]]

        v[0] = -v[0]

        v = [math.cos(-theta) * v[0] - math.sin(-theta) * v[1],
             math.cos(-theta) * v[1] + math.sin(-theta) * v[0]]

        if y < (table_size[1] / 2 - other_paddle_frect.size[1]) or y > (table_size[1] / 2 + other_paddle_frect.size[1]):
            y = table_size[1] / 2

        y = get_hit_height(x, y, v[0], v[1], left_paddle_bound, right_paddle_bound, actual_table_height)
        #print("\t",opponent_y, y)
    elif abs(y - (paddle_frect.pos[1] + paddle_frect.size[1] / 2)) < paddle_frect.size[1]:

        best_d, best_score = 0, 0

        for d in range(-paddle_frect.size[1]//2, paddle_frect.size[1]//2):
            center = y
            rel_dist_from_c = (d / paddle_frect.size[1])
            rel_dist_from_c = min(0.5, rel_dist_from_c)
            rel_dist_from_c = max(-0.5, rel_dist_from_c)
            sign = -1*direction

            theta = sign * rel_dist_from_c * 45 * math.pi / 180

            v = list(ball_speed)

            v = [math.cos(theta) * v[0] - math.sin(theta) * v[1],
                 math.sin(theta) * v[0] + math.cos(theta) * v[1]]

            v[0] = -v[0]

            v = [math.cos(-theta) * v[0] - math.sin(-theta) * v[1],
                 math.cos(-theta) * v[1] + math.sin(-theta) * v[0]]

            x = paddle_frect.pos[0]
            if ball_speed[0] < 0:
                x = paddle_frect.pos[0] + paddle_frect.size[0]

            hit_height = get_hit_height(x, y+d, v[0], v[1], left_paddle_bound, right_paddle_bound, actual_table_height)
            score = abs(hit_height - other_paddle_frect.pos[1] - other_paddle_frect.size[1]) * abs(v[0])

            if score > best_score and v[0] * ball_speed[0] < 0:
                best_d, best_score = d, score


        #print(y - chase_ball(paddle_frect, ball_frect, ball_speed, table_size))

        goal_y = 2

        if other_paddle_frect.pos[1] + (other_paddle_frect.size[1]/2) < table_size[1]/2:
            goal_y = actual_table_height - 2

        dy = goal_y - y

        dx = -(table_size[0]) * direction


        sigma = math.atan2(dy, dx)

        beta = math.atan2(v[1],v[0])
        theta1 = -(sigma-beta)/2
        theta2 = (math.pi-beta-sigma)/2

        d = paddle_frect.size[1] * (4 * theta2) / (math.pi * direction)




        center = paddle_frect.pos[1] + paddle_frect.size[1] / 2
        rel_dist_from_c = (d) / paddle_frect.size[1]
        sign = 1 * direction

        theta = sign * rel_dist_from_c * 45 * math.pi / 180

        v = [ball_speed[0], ball_speed[1] * (-1) ** (hit_height // actual_table_height)]

        v = [math.cos(theta) * v[0] - math.sin(theta) * v[1],
             math.sin(theta) * v[0] + math.cos(theta) * v[1]]

        v[0] = -v[0]

        v = [math.cos(-theta) * v[0] - math.sin(-theta) * v[1],
             math.cos(-theta) * v[1] + math.sin(-theta) * v[0]]

        #print(sigma, math.atan2(v[1],v[0]))




        if abs(d) > abs(paddle_frect.size[1] / 3):
            d = math.copysign(1, d) * paddle_frect.size[1] / 3
        y += best_d
        #print(d, best_d)

        if abs(y - (paddle_frect.pos[1] + paddle_frect.size[1] / 2)) < paddle_frect.size[1] / 3:
            pass#print(y - (paddle_frect.pos[1] + paddle_frect.size[1] / 2))

    prev_ball_pos = ball_frect.pos
    prev_ball_speed = ball_speed

    #return PongAIvAI.directions_from_input(paddle_frect, other_paddle_frect, ball_frect, table_size)

    if paddle_frect.pos[1] + paddle_frect.size[1] / 2 < y + ball_frect.size[1] / 2:
        return "down"
    else:
        return "up"
