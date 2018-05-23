#! /usr/bin/env python
import pygame
from pygame.locals import*
import sys
from pyenv import Robot, FallError
import datetime
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

def walking(x,y):
    X = y
    Y = x
    period = 8
    robot.Walk(0,0,round(-(26*X)), period, round(-(13*Y)))
    #print 'walk(0,0,'+str(round(-(26*y)))+','+str(period)+','+str(round(13*x))

def main():
	pygame.init()
	#screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	#pygame.display.set_caption('Joystick')
	#pygame.display.flip()
        period = 8
        
        while 1:
            for e in pygame.event.get():
                if e.type == pygame.locals.JOYAXISMOTION:
                    x, y = j.get_axis(0), j.get_axis(1)
                    if (x == 0 and y== 0):
                        robot.Cancel()
                    else:
                        walking(x,y)
                        #print '(x, y) = (' + str(float(x)) +', ' + str(float(y))+')'
                        if getBallAxis(robot) == 0:
                            log = str(-1) + ',' + str(-1) + ',' + str(x) + ',' + str(y)+'\n'
                            f.write(log)
                        else :
                            ball = robot.GetLocalPos(robot.HLOBJECT_BALL, robot.HLCOLOR_BALL)
                            b_x, b_y, b_theta = ball[0]
                            log = str(b_x) + ',' + str(b_y) + ',' + str(x) + ',' + str(y) + '\n'
                            f.write(log)

                elif e.type == 11:
                    return -1

if __name__ == '__main__':
    pygame.joystick.init()
    f = open('log' + getDateTime() + '.csv', 'w')
    if pygame.joystick.get_count():
            j = pygame.joystick.Joystick(0)
            j.init()
            print 'Joystick name is' + j.get_name()
            print 'Button number is' + str(j.get_numbuttons())
    else:
            print 'not found Joystick'
            sys.exit()
    robot = Robot()
    main()
    robot.terminate()
    
