#! /usr/bin/env python
import sys
import datetime
import tensorflow as tf
import numpy as np
import kid.strategy
import rcl
import time
import tools.geometry

g_goal0 = [[4.8,1.3,0],[4.8,-1.3,0]]

config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            #per_process_gpu_memory_fraction=0.01
            allow_growth=True
            )
        )

IN_size = 4
TAR_size = 3
INPUT = tf.placeholder(shape=[None, IN_size], dtype=tf.float32)
TARGET = tf.placeholder(shape=[None, TAR_size], dtype=tf.float32)
hidden_size = [30,8]
batch_size = 100
iter_size = 50000

weights = []
biases = []
layers = []
tmp_size = IN_size
last_layer = INPUT
for hsize in hidden_size:
    weights.append(tf.Variable(tf.random_normal(shape=[tmp_size, hsize], stddev=10.0)))
    biases.append(tf.Variable(tf.random_normal(shape=[hsize], stddev=10.0)))
    layers.append(tf.nn.relu(tf.add(tf.matmul(last_layer, weights[-1]), biases[-1]))) 
    tmp_size = hsize
    last_layer = layers[-1]
weights.append(tf.Variable(tf.random_normal(shape=[tmp_size, TAR_size], stddev=10.0)))
biases.append(tf.Variable(tf.random_normal(shape=[TAR_size], stddev=10.0)))
layers.append(tf.add(tf.matmul(last_layer, weights[-1]), biases[-1]))
final_output = layers[-1]
loss = tf.losses.mean_squared_error(TARGET,final_output)


saver = tf.train.Saver()
p_sess = tf.Session(config=config)
ckpt = tf.train.get_checkpoint_state('./w/omo')
if ckpt == None:
        print('!!!!!!!WEIGHTS NOT FOUND!!!!!!')
saver.restore(p_sess, './w/omo/model.ckpt')

def getBallAxis(robot):
    ball = robot.GetLocalPos(robot.HLOBJECT_BALL, robot.HLCOLOR_BALL)
    if not ball:
        return 0
    else:
        return 1

def getDateTime():
    d = datetime.datetime.today()
    date = '{}{}{}{}{}'.format(d.year, '%02d' % d.month, '%02d' % d.day, '%02d' % d.hour, '%02d' % d.minute)
    return date

def walking(x,y,th):
    X = y
    Y = x
    period = 10
    s = 16
    agent.effector.walk(0,round(-(s*th)),round(-(s*X)), period, round(-(s*Y)))

def main():

        while 1:
            print('start')
            g_ball = agent.brain.get_sim_ballpos()
            g_bx = g_ball[0]
            g_by = g_ball[1]
            
            g_selfpos = agent.brain.get_sim_selfpos()
            g_px  = g_selfpos[0]
            g_py  = g_selfpos[1]
            g_pth = g_selfpos[2]

            g_p0x = g_goal0[0][0]
            g_p0y = g_goal0[0][1]
            g_p1x = g_goal0[1][0]
            g_p1y = g_goal0[1][1]
            l_ball = tools.geometry.coord_trans_global_to_local(g_selfpos,g_ball)
            l_pole0 = tools.geometry.coord_trans_global_to_local(g_selfpos,g_goal0[0])
            l_pole1 = tools.geometry.coord_trans_global_to_local(g_selfpos,g_goal0[1])

            agent.wait_until_status_updated()
            agent.brain.dispose_global_position(False)
            agent.brain.start_memorize_observation()
            agent.brain.memorize_visible_observation()
            agent.brain.use_memorized_observation_ntimes(10, 500, 10, 95)
            #agent.brain.sleep(5.0)
            d_ball = agent.brain.get_estimated_object_pos_lc(agent.brain.BALL, agent.brain.AF_ANY)
            d_goal = agent.brain.get_estimated_object_pos_lc(agent.brain.GOAL_POLE, agent.brain.AF_ANY)
            #ball_buf = agent.brain.get_estimated_object_pos_lc(agent.brain.BALL, agent.brain.AF_ANY)
            print('if no mae')
            if (l_ball and (l_pole0 and l_pole1)) :
                print('prin')
                #dbx, dby = d_ball[0]
                #dgx, dgy = d_goal[0]
                bx, by, _ = l_ball
                p0x, p0y, _ = l_pole0
                p1x, p1y, _ = l_pole1
                input_d =np.array([[bx,by,(p0x+p1x)/2.0,(p0y+p1y)/2.0]])
                #print(input_d)
                pred = p_sess.run(final_output, feed_dict={INPUT: input_d})
                x = pred[0][0]
                y = pred[0][1]
                th = pred[0][2]
                print('pred(x,y,th)='+str(x)+','+str(y)+','+str(th))
                if (x == 0 and y== 0):
                    agent.brain.wait_until_motion_finished()
                else:
                    walking(x,y,th)
                print('prin end')
            else:
                print('Ball not find')
            time.sleep(1)
if __name__ == '__main__':
    strategy = kid.strategy.HLKidStrategy()
    agent = rcl.SoccerAgent(lambda: strategy.create_field_properties())
    time.sleep(1)

    #agent.brain.debug_log_ln("cgi.py start")

    agent.brain.set_selfpos((0, 0, 0))

    agent.effector.set_pan_deg(0)

    agent.brain.set_auto_localization_mode(0)
    agent.brain.set_use_white_lines(1)
    agent.brain.enable_auto_wakeup(0)
    #agent.brain.set_use_yolo(1)

    try:
        main()
    except Exception, e:
        agent.brain.debug_log_ln("Exception: " + str(e))
        agent.terminate()
        raise
    except (KeyboardInterrupt, SystemExit):
        agent.terminate()
        raise
    
