import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
import pygame

class FooEnv(gym.Env):
    """
      Observation:
          Type: Box(4)
          Num	Observation                 Min         Max
          0	Box y-Position               0           500
          1	X-Leading Tubes             550          -50
          2	Frst-Tube Height            0            150
          3	Scnd-Tube Height            300          500
          4 X-Back tube                 600          0

          5   Velocity                 -20           20

      Actions:
          Type: Discrete(2)
          Num	Action
          0	Box Not jump
          1	Box jump
    """


    metadata = {
        'render.modes': ['human','rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.reward=0
        self.gravity = -1
        self.boxYPosition = 0
        self.xLeadMin = 550
        self.frstTubeHeightMin = 0
        self.scndTubeHeightMin = 300
        self.xBackTubeMin=600

        self.boxYPositionMax = 500
        self.xLeadMax = -50
        self.frstTubeHeightMax = 150
        self.scndTubeHeightMax = 500
        self.velocityMin= -20
        self.velocityMax= 20
        self.xBackTubeMax=0

        self.boxWidth = 50
        self.boxHeight = 50
        self.boxXPosition = 50


        # valid actions are either (0,1) jump or not jump
        self.action_space = spaces.Discrete(2)
        # recording-> 19 minutes in (low np.array[] and high np.array[])
        # dtype not to float but Integer
        self.low = np.array([self.boxYPosition, self.xLeadMin,self.frstTubeHeightMin, self.scndTubeHeightMin,
                             self.velocityMin,self.xBackTubeMin])
        self.high = np.array([self.boxYPositionMax, self.xLeadMax,self.frstTubeHeightMax,
                              self.scndTubeHeightMax,self.velocityMax,self.xBackTubeMax])

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int)

        # initialization state (same seed= same pattern of numbers)
        self.seed()
        self.state = None
        self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        # use these variables to do stuff and then put them back to state
        boxY, xLead, yTube1,yTube2,velocity,xBackTube = state


        # update the speed and y position of the box

        velocity += self.gravity
        boxY += velocity
        done = False
        # initialise the action JUMP
        if action == 1:
            velocity = 1
            boxY += velocity

        # Moving the Tubes
        xLead=xLead-5
        xBackTube=xBackTube-5

        # Award reward when the tubes have passed the box
        if xLead == -50:
            self.reward += 5.0

        # transfering the tube
        if xLead == -100:
            print("tube back")
            xLead = 550
            xBackTube=650
            yTube1=random.randrange(70, 150)
            yTube2=random.randrange(280, 500)


        # terminate when box is out of bounds
        if boxY-50 < 0 or boxY > 500:
            print("box out of bounds")
            self.reward -= 1.5
            done = True

        # terminate when box collides with tubes
        if 0 <= xLead <= 100 :
            if (boxY > yTube2) or (boxY-50<yTube1):
                print("tube hit")
                self.reward -= 1.5
                done = True

        if not done:
            self.reward += 0.05

        self.state = (boxY, xLead, yTube1, yTube2, velocity, xBackTube)

        return np.array(self.state), self.reward, done, {}




    def reset(self):
        low = np.array([200, 500, 70, 280,0,600])
        high = np.array([450, 500, 150, 500,0,600])

        self.state = self.np_random.uniform(low, high, size=(6,))
        return np.array(self.state)

    def render(self, mode='human', close=False):
        screen_width = 500
        screen_height = 500


        state = self.state
        boxY, xLead, yTube1, yTube2, velocity, xBackTube = state

        self.boxWidth = 50
        self.boxHeight = boxY-50

        self.tubesWidth=xBackTube - xLead

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = 50, self.boxWidth + 50, self.boxHeight - 50, boxY
            box = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.boxtrans = rendering.Transform()
            box.add_attr(self.boxtrans)
            box.set_color(.5, .5, .8)
            self.viewer.add_geom(box)

            l, r, t, b = xBackTube, xLead, yTube1, 0
            tube1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.tube1trans = rendering.Transform()
            tube1.add_attr(self.tube1trans)
            self.viewer.add_geom(tube1)

            l, r, t, b = xBackTube, xLead, yTube2, 600
            tube2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.tube2trans = rendering.Transform()
            tube2.add_attr(self.tube2trans)
            self.viewer.add_geom(tube2)

            self.tube1_geom=tube1
            self.tube2_geom=tube2
            self.box_geom=box


        # tubeDown
        tube1 = self.tube1_geom
        l, r, t, b = xBackTube, xLead, yTube1, 0
        tube1.v =([(l, b), (l, t), (r, t), (r, b)])
        # tubeUp
        tube2 = self.tube2_geom
        l, r, t, b = xBackTube, xLead, yTube2, 600
        tube2.v = ([(l, b), (l, t), (r, t), (r, b)])
        # box
        box = self.box_geom
        l, r, t, b = 50, self.boxWidth + 50, boxY , self.boxHeight
        box.v = ([(l, b), (l, t), (r, t), (r, b)])


        if self.state is None: return None
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

