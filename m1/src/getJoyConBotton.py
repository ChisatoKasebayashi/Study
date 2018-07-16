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

def walking(x,y,th):
    X = y
    Y = x
    period = 8
    robot.Walk(0,round(th*15),round(-(26*X)), period, round(-(13*Y)))
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
                if e.type == pygame.locals.JOYAXISMOTION:
                    x, y = j.get_axis(0), j.get_axis(1)
                    #print('******LEFT*******')
                    #print(str(x)+ ','+str(y))
                    th, ty = j.get_axis(2), j.get_axis(3)
                    #print('********RIGHT*********')
                    #print(str(th)+ ','+str(y))
                    
                    if ((x == 0 and y== 0)and th == 0):
                        robot.Cancel()
                    else:
                        walking(x,y,th)
                        #print('theta = ' + str(th))
                        #print '(x, y) = (' + str(float(x)) +', ' + str(float(y))+')'
                        if getBallAxis(robot) == 0:
                            log = str(-1) + ',' + str(-1) + ',' + str(x) + ',' + str(y)+ ',' + str(th) + ',' + str(ty)+'\n'
                            f.write(log)
                        else :
                            ball = robot.GetLocalPos(robot.HLOBJECT_BALL, robot.HLCOLOR_BALL)
                            b_x, b_y, b_theta = ball[0]
                            log = str(b_x) + ',' + str(b_y) + ',' + str(x) + ',' + str(y) + ',' + str(th) + ',' + str(ty) +'\n'
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
                    robot.Cancel()
if __name__ == '__main__':
    pygame.joystick.init()
    f = open('log' + getDateTime() + '.csv', 'w')
    if pygame.joystick.get_count():
            j = pygame.joystick.Joystick(0)
            j.init()
    else:
            print 'not found Joystick'
            sys.exit()
    robot = Robot()
    main()
    robot.terminate()
    
