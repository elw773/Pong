_C=None
_B=True
_A=False
import threading,time
from urllib import request
import pygame,math,json
thread_running=_A
ai=_C
ai_running=_A
paddle_orientation=_C
prev_move_to_y=0
move_to_y=140
ball_to_y=140
table_size=440,280
paddle_size=10,70
ball_size=15,15
aim_list=[]
towards_paddle=_A
paddle_speed=1
ball_x_vel=0
he_ded=_A
my_paddle=_C
hit_avg=35
class game_ai:
        def __init__(A,orientation,ball_pos):B=ball_pos;A.paddle_orientation=orientation;A.prev_ball_pos=B;A.ball_pos=B;A.ball_info=[0,0,0,0];A.prev_ball_vel=[0,0];A.ball_vel=[0,0];A.ball_direction=0;A.max_angle=45;A.calculated=_A;A.enemy_calculated=_A;A.ball_info=[0,0,0,0];A.max_loop=2000;A.hit_tracker=_B;A.prev_endpoint=[];A.score_checked=_A;A.score=[0,0];A.prev_predicted_endpoint=-1000000000.0;A.prev_enemy_pos=[0,0]
        def wall_collision(B,pos_y):A=pos_y;return int(A-0.001)<0 or int(A+0.001)+ball_size[1]>table_size[1]
        def paddle_collision(B,pos_x):
                A=pos_x
                if int(A+0.001)>24 and int(A-0.01)<table_size[0]-25-ball_size[0]:return _A
                else:return _B
        def check_win(B,pos_x):
                A=pos_x
                if A<15:return-1
                elif A>425:return 1
        def skip_frame(J,pos_x,pos_y,vel_x,vel_y,move_factor):
                I=pos_y;H=pos_x;D=vel_y;C=vel_x;A=move_factor
                try:
                        B=0;E=0;F=0
                        if D<0:B=I+1;E=int(B/(-D*A)+0.999999)
                        else:B=abs(266-I);E=int(B/(D*A)+0.999999)
                        if C<0:G=abs(H-25);F=int(G/(-C*A)+0.999999)
                        else:G=abs(H-400);F=int(G/(C*A)+0.999999)
                        return min(E,F)
                except:return 1000000000.0
        def get_ball_endpoint(A,pos_x,pos_y,vel_x,vel_y):
                G=vel_x;F=pos_x;C=vel_y;B=pos_y;E=0;I=int((G**2+C**2)**0.5);D=1
                if I>0:D=1.0/I
                while not A.paddle_collision(F):
                        E+=1
                        if E>A.max_loop:break
                        if A.wall_collision(B):
                                H=0
                                while A.wall_collision(B):
                                        E+=1;F+=-0.1*G*D;B+=-0.1*C*D;H+=1
                                        if E>A.max_loop:break
                                C=-C
                                while H>0 or A.wall_collision(B):
                                        E+=1;F+=0.1*G*D;B+=0.1*C*D;H-=1
                                        if E>A.max_loop:break
                        else:J=A.skip_frame(F,B,G,C,D);F+=G*D*J;B+=C*D*J
                if E>A.max_loop:0
                return F,B,G,C
        def get_angle(C,paddle_y,ball_y,p_orientation):
                D=paddle_y+paddle_size[1]/2;A=(ball_y-D)/paddle_size[1];A=min(0.5,A);A=max(-0.5,A);B=0
                if p_orientation==1:B=1
                E=1-2*B;return E*A*C.max_angle*math.pi/180
        def get_paddle_collision(B,pos_x,pos_y,vel_x,vel_y,paddle_loc_y,move_factor,p_orientation):
                L=p_orientation;I=move_factor;G=vel_y;F=vel_x;E=pos_x;D=pos_y;J=0;H=0
                while B.paddle_collision(E)and not B.wall_collision(D):
                        H+=1;E-=0.1*F*I;D-=0.1*G*I;J+=1
                        if H>B.max_loop:break
                C=B.get_angle(paddle_loc_y,D+0.5*ball_size[1],L);A=[F,G];A=[math.cos(C)*A[0]-math.sin(C)*A[1],math.sin(C)*A[0]+math.cos(C)*A[1]];A[0]=-A[0];A=[math.cos(-C)*A[0]-math.sin(-C)*A[1],math.cos(-C)*A[1]+math.sin(-C)*A[0]];K=0
                if L==1:K=1
                try:
                        if A[0]*(2*K-1)<1:A[1]=A[1]/abs(A[1])*math.sqrt(A[0]**2+A[1]**2-1);A[0]=2*K-1
                except:pass
                F=A[0];G=A[1]
                while J or B.paddle_collision(E)and not B.wall_collision(D):
                        H+=1;E+=0.1*F*I;D+=0.1*G*I;J-=1
                        if H>B.max_loop:break
                if H>B.max_loop:0
                return E,D,F,G
        def calc_hits(D,pos_x,pos_y,vel_x,vel_y,enemy):
                H=enemy;G=pos_y;E=vel_y;A=vel_x;M=time.time();aim_list.clear();I=int((A**2+E**2)**0.5);J=1
                if I>0:J=1.0/I
                for L in range(1,paddle_size[1]-1,1):
                        B=G-L
                        if B>=0 and B<=table_size[1]-paddle_size[1]:
                                K=D.paddle_orientation
                                if H:K*=-1
                                C=D.get_paddle_collision(pos_x,G,A,E,B,J,K);F=D.get_ball_endpoint(C[0],C[1],C[2],C[3])
                                if not H:aim_list.append((F[1],B,A,E,F[0]))
                                else:aim_list.append((F[1]-paddle_size[1]/2-ball_size[1]/2,A,0,0,0))
        def calc(A):global move_to_y,ball_to_y;B=time.time();A.ball_info=A.get_ball_endpoint(A.ball_pos[0],A.ball_pos[1],A.ball_vel[0],A.ball_vel[1]);move_to_y=A.ball_info[1];ball_to_y=A.ball_info[1];A.calc_hits(A.ball_info[0],A.ball_info[1],A.ball_info[2],A.ball_info[3],enemy=_A)
        def enemy_calc(A):B=time.time();A.ball_info=A.get_ball_endpoint(A.ball_pos[0],A.ball_pos[1],A.ball_vel[0],A.ball_vel[1]);A.calc_hits(A.ball_info[0],A.ball_info[1],A.ball_info[2],A.ball_info[3],enemy=_B);'\n        sum_total = 0\n        cnt = 0\n        for aim in aim_list:\n            cnt += 1\n            sum_total += min(table_size[1] - paddle_size[1], aim[0] + paddle_size[1] / 2)\n        move_to_y_test = sum_total / cnt\n        '
        def paddle_dis_to_ball(E,ball_y,paddle_y):
                C=paddle_y;B=ball_y;A=C-(B+ball_size[1]);D=B-(C+paddle_size[1])
                if A<=0 and D<=0:return A
                else:return min(abs(A),abs(D))
        def update_score(A,pos_x):
                B=pos_x
                if A.check_win(B)and not A.score_checked:
                        if A.check_win(B)==A.paddle_orientation:A.score[0]+=1
                        else:A.score[1]+=1
                        A.score_checked=_B
        def towards_paddle(A):
                if A.ball_vel[0]!=0:
                        A.ball_direction=int(A.ball_vel[0]/abs(A.ball_vel[0]))
                        if A.ball_direction!=A.paddle_orientation:return _B
                        else:return _A
                else:return _A
        def get_predicted_course(A,enemy_pos_y,move_factor,calc_range=[0,0]):
                B=calc_range;D=0
                for E in range(B[0],B[1]+1):C=A.get_paddle_collision(A.ball_info[0],A.ball_info[1],A.ball_info[2],A.ball_info[3],enemy_pos_y+E,move_factor,A.paddle_orientation*-1);F=A.get_ball_endpoint(C[0],C[1],C[2],C[3]);D+=F[1]
                D/=B[1]-B[0]+1;return D
        def predict_enemy_hit(A,enemy_pos_y):
                B=enemy_pos_y;global move_to_y;E=A.paddle_dis_to_ball(A.ball_info[1],B)*-1;C=int((A.ball_info[2]**2+A.ball_info[3]**2)**0.5);D=1
                if C>0:D=1.0/C
                move_to_y=A.get_predicted_course(B,D)
        def update(A,ball_pos,enemy_pos):
                D=enemy_pos;C=ball_pos;global ball_to_y,move_to_y,paddle_orientation,towards_paddle,ball_x_vel,he_ded
                if abs(A.prev_ball_pos[0]-C[0])>100:A.calculated=_A;A.enemy_calculated=_A;A.score_checked=_A
                A.prev_ball_pos=A.ball_pos;A.ball_pos=C;A.paddle_orientation=paddle_orientation;A.prev_ball_vel=A.ball_vel;A.ball_vel=[A.ball_pos[0]-A.prev_ball_pos[0],A.ball_pos[1]-A.prev_ball_pos[1]];ball_x_vel=A.ball_vel[0];towards_paddle=A.towards_paddle()
                if not A.paddle_collision(A.ball_pos[0]):
                        A.hit_tracker=_B
                        if towards_paddle:
                                he_ded=_A
                                if not A.calculated:
                                        B=A.get_ball_endpoint(A.ball_pos[0],A.ball_pos[1],A.ball_vel[0],A.ball_vel[1])
                                        if A.prev_endpoint==B:A.calculated=_B;A.enemy_calculated=_A;threading.Thread(target=A.calc).start()
                                        elif A.calculated==_A:A.prev_endpoint=B
                        else:
                                A.predict_enemy_hit(D[1]);E=-ball_size[0]
                                if A.paddle_orientation*-1==1:E=-paddle_size[0]
                                F=abs(C[0]-D[0])+E
                                if ball_x_vel!=0:
                                        G=abs(F/ball_x_vel);H=A.paddle_dis_to_ball(A.ball_info[1],D[1])
                                        if G<H-10:he_ded=_B
                                        else:he_ded=_A
                                else:0
                                if not A.enemy_calculated:
                                        B=A.get_ball_endpoint(A.ball_pos[0],A.ball_pos[1],A.ball_vel[0],A.ball_vel[1])
                                        if A.prev_endpoint==B:A.calculated=_A;A.enemy_calculated=_B;threading.Thread(target=A.enemy_calc).start()
                                        elif not A.enemy_calculated:A.prev_endpoint=B
def pong_ai(paddle_frect,other_paddle_frect,ball_frect,table_size):
        D=other_paddle_frect;C=paddle_frect;A=ball_frect;global ai,paddle_orientation,ai_running,move_to_y,ball_to_y,towards_paddle,paddle_speed,ball_x_vel;global client_thread,kill,old_opponent_code,old_render_code,scratch,scratch_executed;global first_run,opponent_function,hax_thread,my_paddle
        if C.pos[0]<D.pos[0]:paddle_orientation=1
        else:paddle_orientation=-1
        if not ai_running or paddle_orientation!=ai.paddle_orientation:ai_running=_B;ai=game_ai(paddle_orientation,[A.pos[0],A.pos[1]])
        ai.update([A.pos[0],A.pos[1]],D.pos);E=0;F=D.pos[1],D.pos[1]+paddle_size[1]
        if towards_paddle:
                if len(aim_list)>0:
                        G=-A.size[0]
                        if paddle_orientation==1:G=-C.size[0]
                        I=abs(A.pos[0]-C.pos[0])+G
                        for (J,B) in enumerate(aim_list):
                                H=min(abs(B[0]+ball_size[1]-F[0]),abs(B[0]-F[1]))
                                if abs(H)>E:
                                        if B[1]-ball_size[1]+2<ball_to_y and B[1]+paddle_size[1]-2>ball_to_y:E=abs(H);move_to_y=B[1]
                        if E==0:move_to_y=ball_to_y-paddle_size[1]/2
        else:0
        if he_ded:move_to_y=105
        if C.pos[1]<move_to_y:return'down'
        else:return'up'