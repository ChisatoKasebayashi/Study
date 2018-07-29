#! /usr/bin/env python
import pygame
from pygame.locals import*
import sys
import datetime
import rcl
import kid.strategy
import time
import tools.geometry

g_goal0 = [[4.8, 1.3,0], [4.8, -1.3,0]]
def getBallAxis(robot):
    ball = agent.brain.get_sim_selfpos()
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
    period = 8
    agent.effector.walk(0,round(-(16*th)),round(-(16*X)), period, round(-(16*Y)))

def main():
	pygame.init()
        print('|----------------------------------------------------------|')
        print('|                    HOW TO USE                            |')
        print('|----------------------------------------------------------|')
        print('| Left Analog Stick   : MOVE      foward back right left   |')
        print('| Right Analog Stick  : TURN  <-clockwise  anticlockwise-> |')
        print('| Button 1            : CLOSE                              |')
        print('| Button push >= 5    : Fall Error flag                    |')
        print('|----------------------------------------------------------|')
        period = 8
        
        while 1:
            for e in pygame.event.get():
                if e.type == pygame.locals.JOYAXISMOTION:
                    jx, jy = j.get_axis(0), j.get_axis(1)
                    jth, jty = j.get_axis(2), j.get_axis(3)
                    

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

                    if ((jx == 0 and jy== 0)and jth == 0):
                        agent.effector.cancel()

                    else:
                        walking(jx,jy,jth)
                        if getBallAxis(robot) == 0:
                            log = str(-1)+','+ str(-1)+','+str(l_pole0[0])+','+str(l_pole0[1])+ ','+ str(l_pole1[0])+ ','+str(l_pole1[1]) + ',' +str(g_bx) +','+str(g_by)+','+str(g_p0x) +',' +str(g_p0y)+','+str(g_p1x)+','+str(g_p1y) +',' +  str(g_px) + ',' + str(g_py) + ',' + str(g_pth) + ',' + str(jx) + ',' + str(jy) + ',' + str(jth) + ',' + str(jty) +'\n'
                            f.write(log)
                        else :
                            log = str(l_ball[0])+','+ str(l_ball[1])+','+str(l_pole0[0])+','+str(l_pole0[1])+ ','+ str(l_pole1[0])+ ','+str(l_pole1[1]) + ',' +str(g_bx) +','+str(g_by)+','+str(g_p0x) +',' +str(g_p0y)+','+str(g_p1x)+','+str(g_p1y) +',' +  str(g_px) + ',' + str(g_py) + ',' + str(g_pth) + ',' + str(jx) + ',' + str(jy) + ',' + str(jth) + ',' + str(jty) +'\n'
                            f.write(log)
                elif e.type == pygame.locals.JOYBUTTONDOWN:
                    print str(e.button)+'th button'
                    if e.button == 0: # END
                        print("CLOSE")
                        f.close()
                        return -1
                    elif e.button >=  4:
                        print '!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!'
                        log = 'fallerror\n'
                        f.write(log)
                elif e.type == pygame.locals.JOYBUTTONUP:
                    agent.effector.cancel()
                    #robot.Cancel()
if __name__ == '__main__':
    strategy = kid.strategy.HLKidStrategy()
    pygame.joystick.init()
    f = open('log' + getDateTime() + '.csv', 'w')
    if pygame.joystick.get_count():
            j = pygame.joystick.Joystick(0)
            j.init()
    else:
            print 'not found Joystick'
            sys.exit()
    agent = rcl.SoccerAgent(lambda: strategy.create_field_properties())
    time.sleep(1)

    agent.brain.debug_log_ln("cgi.py start")

    agent.brain.set_selfpos((0, 0, 0))

    agent.effector.set_pan_deg(0)

    agent.brain.set_auto_localization_mode(0)
    agent.brain.set_use_white_lines(1)
    agent.brain.enable_auto_wakeup(0)
    agent.brain.set_use_yolo(1)
    robot = agent.brain
    try:
        main()
    except Exception, e:
        agent.brain.debug_log_ln("Exception: " + str(e))
        agent.terminate()
        raise
    except (KeyboardInterrupt, SystemExit):
        agent.terminate()
        raise
    
