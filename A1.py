#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 00:55:18 2018

@author: siyanluo
"""
import sys
import numpy as np

NO_OBSERVATION = -1
HAS_OBSERVATION = -2

# Constance
NUM_TERMINAL_STATE = 2
NUM_WALL = 1

TYPE_TERMINAL_STATE = 'terminal'
TYPE_WALL = 'wall'
TYPE_NORMAL = 'normal'

ONE_WALL = 'one'
TWO_WALL = 'two'
END = 'end'

# Random values are assigned to directions,
# MUST satisfy:
# UP = -DOWN
# LEFT = -RIGHT
UP = 10
DOWN = -10
LEFT = 20
RIGHT = -20

# States (12 states = 9 non-terminal + 2 terminal):
state11 = {'x':1, 'y':1, 'value':0.0, 'type': TYPE_NORMAL}
state12 = {'x':1, 'y':2, 'value':0.0, 'type': TYPE_NORMAL}
state13 = {'x':1, 'y':3, 'value':0.0, 'type': TYPE_NORMAL}
state14 = {'x':1, 'y':4, 'value':0.0, 'type': TYPE_NORMAL }
state21 = {'x':2, 'y':1, 'value':0.0, 'type': TYPE_NORMAL }
state22 = {'x':2, 'y':2, 'value':0.0, 'type': TYPE_WALL }
state23 = {'x':2, 'y':3, 'value':0.0, 'type': TYPE_NORMAL}
state24 = {'x':2, 'y':4, 'value':0.0, 'type': TYPE_TERMINAL_STATE}
state31 = {'x':3, 'y':1, 'value':0.0, 'type': TYPE_NORMAL }
state32 = {'x':3, 'y':2, 'value':0.0, 'type': TYPE_NORMAL }
state33 = {'x':3, 'y':3, 'value':0.0, 'type': TYPE_NORMAL }
state34 = {'x':3, 'y':4, 'value':0.0, 'type': TYPE_TERMINAL_STATE }


# P(s'|s, a) - the probability of performing action a to make agent move from 
# state to state prime
P_CORRECT_DIR = 0.8
P_OPPOSITE_DIR = 0.0
P_OTHER_DIR = 0.1

# Grid:
#-----------\-----------\----------\---------------------
#           \           \          \                       \
#    s31    \  s32      \  s33     \    s34(terminal, +1)  \
#-----------\------------\----------\----------------------
#           \            \          \                       \
#   s21     \ s22 (WALL) \   s23    \   s24(terminal, -1)   \
#-----------\------------\----------\-------------------------
#           \            \          \                   \
#    s11    \ s12        \  s13     \   s14             \
#-----------\------------\--------- \--------------------
#
#

grid = [[state31, state32, state33, state34],
        [state21, state22, state23, state24],
        [state11, state12, state13, state14]]

# Initialize all non-terminal, non-wall states
def initGrid(hasInitialState, initState):
    if hasInitialState:
        initProb = 0.0
        initState['value'] = 1.0
        print("Custom Initial State:", initState)
    else:
        initProb = round(1/float(len(grid)*len(grid[0]) - NUM_TERMINAL_STATE - NUM_WALL),3)
        for row in range(len(grid)):
            for state in grid[row]:
                if state['type'] == TYPE_NORMAL:
                    state['value'] = initProb

def getState(x, y):
    return grid[3-x][y-1] 

def getAdjacentState(state, direction):
    if state['type'] == TYPE_WALL:
        print("-----ERROR! TYPE_WALL type gets passed in!------")
        return   
    x = state['x']
    y = state['y']
    orginalState = state
    adjacentState = orginalState
    if direction == LEFT and y > 1:
        adjacentState = getState(x, y-1)
    elif direction == RIGHT and y < 4:
        adjacentState = getState(x, y+1)
    elif direction == UP and x < 3:
        adjacentState = getState(x+1, y)
    elif direction == DOWN and x > 1:
        adjacentState = getState(x-1,y)
      
    # Check if the state is WALL
    if adjacentState['type'] == TYPE_WALL:
        adjacentState = orginalState
    return adjacentState
        
def performUpdate(action, observation, mode):
    newVals = np.zeros((4,5))
    # MUST create a new list to store updated values to avoid reading updated value
    # b'(s) from adjacent state
    for row in range(1, len(grid) +1):
        for col in range(1, len(grid[0])+1):
            state = getState(row, col)
            if state['type'] != TYPE_WALL:
                newVals[row][col] = calBelifStateVal(state, observation, action, mode)
     
    for row in range(1, len(grid)+1):
        for col in range(1, len(grid[0])+1):
            state = getState(row, col)
            state['value'] = newVals[row][col]
                
def performSequentialUpdate(actions, observations, mode):
    if len(actions) != len(observations):
        print("-----ERROR! Number of actions should match number of observations! ------- ")
    for i in range(len(actions)):
        performUpdate(actions[i], observations[i], mode)


################ Primary logic for the algirithm ########################
# Parameters:
# state - the current updated state s' (the arrival state)
# observation - observation info at state s'
# action - the intended action the agent is supposed to take
# (Note: not the actual movement as the robot's actual movements are prob distribution)
# mode:
#   NO_OBSERVATION - P(observation = nothing| s') = 1.0
#   HAS_OBSERVATION - P(observation = nothing| s') = 0.0
                
def calBelifStateVal(state, observation, action, mode):
    if mode == NO_OBSERVATION:
        return 1.0 * performAction(state, action)
    else:
        # Observation value should be the observation of s' not s
        # NOTE: The agent moves from s to s'
        return getObservationModelVal(state, observation) * performAction(state, action)

def getObservationModelVal(state, observation):
    if observation == ONE_WALL:
        if (state['x'] == 3 and state['y'] == 4) or (state['x'] == 2 and state['y'] == 4):
            return 0.0
        elif state['y'] == 3:
            return 0.9
        else:
            return 0.1
        
    elif observation == TWO_WALL:
        if (state['x'] == 3 and state['y'] == 4) or (state['x'] == 2 and state['y'] == 4):
            return 0.0
        elif state['y'] == 3:
            return 0.1
        else:
            return 0.9
        
    elif observation == END:
        if (state['x'] == 3 and state['y'] == 4) or (state['x'] == 2 and state['y'] == 4):
            return 1.0
        else:
            return 0.0   

def performAction(state, action):
    probSum = 0.0
    probSum = getProbbyActionDir(state, action, DOWN) + \
              getProbbyActionDir(state, action, LEFT) + \
              getProbbyActionDir(state, action, UP)+\
              getProbbyActionDir(state, action, RIGHT)
 
    return probSum

# Get the probability P(s'|s,A) * b(s) by actual movement direction
# oriDir - the agent was told to go oriDir
# actualDir - the agent was actually going actualDir
def getProbbyActionDir(state, oriDir, actualDir):
    dirProb = 0.0
    prob = 0.0 
    prevState = getAdjacentState(state, -actualDir)
    
    # The agent is moving from prevState to state:
    # if the adjacent prevState(s) does not exist (it is a wall),
    # and moving from s to s' in that direction is possible, P(s'|s,a) > 0, 
    # it would hit the wall and bounce back
    
    # the agent moves in the OPPOSITE direction
    if oriDir == -actualDir:
        if prevState == state:
            # hit the wall and bounce back - starting from itself 
            dirProb = P_CORRECT_DIR
        else:
            # no wall - not possible to move in the opposite direction
            dirProb = 0.0
    
    # the agent moves in the EXPECTED direction
    elif oriDir == actualDir:
        # previousState is wall or terminal state - not possible
        if prevState == state or prevState['type'] == TYPE_TERMINAL_STATE:
            dirProb = 0.0
        else:
            dirProb = P_CORRECT_DIR
    else:
        dirProb = P_OTHER_DIR
       
    prob = dirProb * prevState['value']
    return prob
##################################################################

def normalization(grid):
    total = 0.0
    for row in grid:
        for state in row:
            total += state['value']
    
    for row in grid:
        for state in row:
            state['value'] = round(state['value'] / total, 5)

def columnize(word, width, align='Left'):
    nSpaces = width - len(word)
    halfnSpaces = int(nSpaces/2)
    if nSpaces < 0:
        nSpaces = 0
    if align == 'Left':
        return word + (" " * nSpaces)
    if align == 'Right':
        return (" " * nSpaces) + word
    return (" " * (halfnSpaces)) + word + (" " * (nSpaces-halfnSpaces)) + " "


def printGrid(grid):
    column = len(grid[0])
    for i, row in enumerate(grid):           
        print ('%s%s' % (columnize('Row %d |' % (len(grid[0]) - i - 1), column*2, 'Right'), 
                        '|'.join([columnize(str(state['value']), column, 'Center') for state in row])))

def main(args):
    #o (up, up , up) (2,2,2)   
    initGrid(False, {})
    performSequentialUpdate([UP, UP, UP], [TWO_WALL, TWO_WALL, TWO_WALL], HAS_OBSERVATION)
    normalization(grid)
    printGrid(grid)
    print('\n')
    
    #o (up, up, up) (1,1,1)   
    initGrid(False, {})
    performSequentialUpdate([UP, UP, UP], [ONE_WALL, ONE_WALL, ONE_WALL], HAS_OBSERVATION)
    normalization(grid)
    printGrid(grid)
    print('\n')
    
    #o (right, right, up) (1,1,end) with S0 = (2,3)  
    initGrid(True, state32)
    performSequentialUpdate([RIGHT, RIGHT, UP], [ONE_WALL, ONE_WALL, END], HAS_OBSERVATION)
    normalization(grid)
    printGrid(grid)
    print('\n')
    
    #o (up, right, right, right) (2,2,1,1) with S0 = (1,1)
    initGrid(True, state11)
    performSequentialUpdate([UP, RIGHT, RIGHT, RIGHT], [TWO_WALL, TWO_WALL, ONE_WALL, ONE_WALL], HAS_OBSERVATION)
    normalization(grid)
    printGrid(grid)  
    
if __name__ == '__main__':
    main(sys.argv)
