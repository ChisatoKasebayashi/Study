#! /usr/bin/env python

import pygame
from pygame.locals import*

from pyenv import Robot, FallError

#SCREEN_WIDTH = 640
#SCREEN_HEIGHT = 480

pygame.joystick.init()
try:
	j = pygame.joystick.Joystick(0)
	j.init()
	print 'Joystick name is' + j.get_name()
	print 'Button number is' + str(j.get_numbuttons())
except pygame.error:
	print 'not found Joystick'

def main():
	pygame.init()
	#screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	#pygame.display.set_caption('Joystick')
	#pygame.display.flip()
        period = 8
        robot = Robot()

	while 1:
		for e in pygame.event.get():
			if e.type == QUIT:
				return
			if (e.type == KEYDOWN and e.key == K_ESCAPE):
				return
			if e.type == pygame.locals.JOYAXISMOTION:
				x, y = j.get_axis(0), j.get_axis(1)
				print '(x, y) = (' + str(float(x)) +', ' + str(float(y))+')'
                                if (x == 0 and y== 0):
                                    print 'cancel'
                                    robot.Cancel()
                                else:
                                    robot.Walk(0,0,round(-(26*y)), period, round(13*x))
                                    print 'walk(0,0,'+str(round(-(26*y)))+','+str(period)+','+str(round(13*x))
			elif e.type == pygame.locals.JOYBALLMOTION:
				print 'ball motion'
			elif e.type == pygame.locals.JOYHATMOTION:
				print 'hat motion'
			elif e.type == pygame.locals.JOYBUTTONDOWN:
				print str(e.button) + 'th button pushed'
                                #robot.Walk(0,0,0, period, 0)
			elif e.type == pygame.locals.JOYBUTTONUP:
				print str(e.button) + 'th button uped'

if __name__ == '__main__':main()

