#! /usr/bin/env python
import pygame
from pygame.locals import*
import sys
import datetime
import rcl
import kid.strategy
import time
def getBallAxis(robot):
    #ball = robot.GetLocalPos(robot.HLOBJECT_BALL, robot.HLCOLOR_BALL)
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
    #print 'walk(0,0,'+str(round(-(26*y)))+','+str(period)+','+str(round(13*x))

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
                #agent.wait_until_status_updated()
                if e.type == pygame.locals.JOYAXISMOTION:
                    jx, jy = j.get_axis(0), j.get_axis(1)
                    #print('******LEFT*******')
                    #print(str(x)+ ','+str(y))
                    jth, jty = j.get_axis(2), j.get_axis(3)
                    #print('********RIGHT*********')
                    #print(str(th)+ ','+str(y))
                    
                    if ((jx == 0 and jy== 0)and jth == 0):
                        #robot.Cancel()
                        agent.effector.cancel()

                    else:
                        walking(jx,jy,jth)
                        #print('theta = ' + str(th))
                        #print('selfpos')
                        #print(agent.brain.get_sim_selfpos()[0])
                        #print('ballpos')
                        #print(agent.brain.get_sim_ballpos()[0])
                        #print '(x, y) = (' + str(float(x)) +', ' + str(float(y))+')'
                        if getBallAxis(robot) == 0:
                            b_x = -1
                            b_y = -1
                            selfpos = agent.brain.get_sim_selfpos()
                            pos_x  = selfpos[0]
                            pos_y  = selfpos[1]
                            pos_th = selfpos[2]
                            log = str(b_x) + ',' + str(b_y) + ',' + str(pos_x) + ',' + str(pos_y) + ',' + str(pos_th) + ',' + str(jx) + ',' + str(jy) + ',' + str(jth) + ',' + str(jty) +'\n'
                            f.write(log)
                        else :
                            #ball = robot.GetLocalPos(robot.HLOBJECT_BALL, robot.HLCOLOR_BALL)
                            ball = agent.brain.get_sim_ballpos()
                            b_x = ball[0]
                            b_y = ball[1]
                            selfpos = agent.brain.get_sim_selfpos()
                            pos_x  = selfpos[0]
                            pos_y  = selfpos[1]
                            pos_th = selfpos[2]
                            log = str(b_x) + ',' + str(b_y) + ',' + str(pos_x) + ',' + str(pos_y) + ',' + str(pos_th) + ',' + str(jx) + ',' + str(jy) + ',' + str(jth) + ',' + str(jty) +'\n'
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
    
