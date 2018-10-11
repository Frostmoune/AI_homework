# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def evaluationFunction(self, currentGameState, action, agentIndex):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generateSuccessor(agentIndex, action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()
    
    def countFood(self, gameState):
      nowFood = gameState.getFood()
      count = 0
      for x in nowFood:
        for y in x:
          if y:
            count += 1
      return count

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxValue(currentGameState, depth):
          bestScore = -1e10

          if depth >= self.depth or currentGameState.isWin() or currentGameState.isLose():
            score = self.evaluationFunction(currentGameState)
            return score
          
          for action in currentGameState.getLegalActions(0):
            # for ghost in range(1, currentGameState.getNumAgents()):
            nextGameState = currentGameState.generateSuccessor(0, action)
            bestScore = max(bestScore, minValue(nextGameState, depth, 1))

          return bestScore

        def minValue(currentGameState, depth, ghost):
          bestScore = 1e10

          if depth >= self.depth or currentGameState.isWin() or currentGameState.isLose():
            score = self.evaluationFunction(currentGameState)
            return score
          
          for action in currentGameState.getLegalActions(ghost):
            nextGameState = currentGameState.generateSuccessor(ghost, action)
            if ghost < currentGameState.getNumAgents() - 1:
              bestScore = min(bestScore, minValue(nextGameState, depth, ghost + 1))
            else:
              bestScore = min(bestScore, maxValue(nextGameState, depth + 1))

          return bestScore

        def minimax(currentGameState):
          legalMoves = currentGameState.getLegalActions(0)
          bestScore = -1e10
          bestAction = 0
          for action in legalMoves:
            # for ghost in range(1, currentGameState.getNumAgents()):
            nextGameState = currentGameState.generateSuccessor(0, action)
            score = minValue(nextGameState, 0, 1)
            if bestScore < score:
              bestScore = score
              bestAction = action

          return (bestScore, bestAction)
        
        # print("Food:", self.countFood(gameState))
        
        now_action = minimax(gameState)

        print "Score:", now_action[0]

        return now_action[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # def alphaBeta(currentGameState, depth, ghostNum, alpha, beta):
        #   if depth >= self.depth or currentGameState.isWin() or currentGameState.isLose():
        #     return self.evaluationFunction(currentGameState)

        #   actions = currentGameState.getLegalActions(ghostNum)
        #   if ghostNum == 0:
        #     bestScore = float("-inf")
        #     for action in actions:
        #       nextGameState = currentGameState.generateSuccessor(0, action)
        #       bestScore = max(bestScore, alphaBeta(nextGameState, depth, 1, alpha, beta))
        #       if bestScore > beta:
        #         return bestScore
        #       alpha = max(alpha, bestScore)
        #     return alpha

        #   elif ghostNum > 0:
        #     bestScore = float("inf")
        #     allGhost = currentGameState.getNumAgents()
        #     for action in actions:
        #       nextGameState = currentGameState.generateSuccessor(ghostNum, action)
        #       if ghostNum == allGhost - 1:
        #         bestScore = min(bestScore, alphaBeta(nextGameState, depth + 1, 0, alpha, beta))
        #       else:
        #         bestScore = min(bestScore, alphaBeta(nextGameState, depth, ghostNum + 1, alpha, beta))
        #       if bestScore < alpha:
        #         return bestScore
        #     return beta

        def maxValue(currentGameState, depth, alpha, beta):
          bestScore = float('-inf')

          if depth >= self.depth or currentGameState.isWin() or currentGameState.isLose():
            score = self.evaluationFunction(currentGameState)
            return score
          
          actions = currentGameState.getLegalActions(0)
          
          for action in actions:
            nextGameState = currentGameState.generateSuccessor(0, action)
            bestScore = max(bestScore, minValue(nextGameState, depth, 1, alpha, beta))

            if bestScore > beta:
              return bestScore

            alpha = max(alpha, bestScore)

          return bestScore

        def minValue(currentGameState, depth, ghost, alpha, beta):
          bestScore = float('inf')
          
          if depth >= self.depth or currentGameState.isWin() or currentGameState.isLose():
            score = self.evaluationFunction(currentGameState)
            return score

          if ghost >= currentGameState.getNumAgents():
            return maxValue(currentGameState, depth + 1, alpha, beta)

          actions = currentGameState.getLegalActions(ghost)
          
          for action in actions:
            nextGameState = currentGameState.generateSuccessor(ghost, action)
            bestScore = min(bestScore, minValue(nextGameState, depth, ghost + 1, alpha, beta))
            
            if bestScore < alpha:
              return bestScore
            beta = min(bestScore, beta)

          return bestScore

        def alphaBeta(currentGameState, alpha, beta):
          legalMoves = currentGameState.getLegalActions(0)
          bestScore = float('-inf')
          bestAction = 'Stop'
          for action in legalMoves:
            nextGameState = currentGameState.generateSuccessor(0, action)
            score = minValue(nextGameState, 0, 1, alpha, beta)

            if bestScore < score:
              bestScore = score
              bestAction = action
            
            if bestScore > beta:
              return (bestScore, bestAction)

            alpha = max(alpha, bestScore)

          return (bestScore, bestAction)
        
        now_action = alphaBeta(gameState, float('-inf'), float('inf'))

        print "Score:", now_action[0]

        return now_action[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

