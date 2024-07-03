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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = 0  
        # o sa initializez scorul la 0

        stare = newGhostStates[0].configuration.pos
        #obtin poz priei fantome din startea viitoare cum ar fi
        closestGhost = manhattanDistance(newPos, closestGhost)
        #calc dist Manhattan intre poz noua si cea mai apropiata fantoma

    
        newFoodPositions = newFood.asList()

      #obtin poz noii mancari

        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]
#calculez dist Manhattan de la pacman la fiecare buc de mancare

        if len(foodDistances) == 0:
            return 0

        closestFood = min(foodDistances)

        
        score -= 50

        return successorGameState.getScore() + closestGhost / (closestFood * 10) + score
       

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
       
        result = self.get_value(gameState, 0, 0)
#aleg cea mai buna actiune folosind algoritmul Minimax
        return result[1]

    def get_value(self, gameState, index, depth):
      
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""
        if index == 0:
            return self.max_value(gameState, index, depth)
        else:
            return self.min_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]

            if current_value < min_value:
                min_value = current_value
                min_action = action

        return min_value, min_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        result = self.getBestActionAndScore(gameState, 0, 0, float("-inf"), float("inf"))
#aleg cea mai buna actiune folosind alg minimax cu taiere alphabeta
        
        return result[0]

    def getBestActionAndScore(self, game_state, index, depth, alpha, beta):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", game_state.getScore()

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(game_state, index, depth, alpha, beta)

        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(game_state, index, depth, alpha, beta)

    def max_value(self, game_state, index, depth, alpha, beta):
        """
        Returns the max utility action-score for max-agent with alpha-beta pruning
        """
        legalMoves = game_state.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Update max_value and max_action for maximizer agent
            if current_value > max_value:
                max_value = current_value
                max_action = action

            # Update alpha value for current maximizer
            alpha = max(alpha, max_value)

            # Pruning: Returns max_value because next possible max_value(s) of maximizer
            # can get worse for beta value of minimizer when coming back up
            if max_value > beta:
                return max_action, max_value

        return max_action, max_value

    def min_value(self, game_state, index, depth, alpha, beta):
        """
        Returns the min utility action-score for min-agent with alpha-beta pruning
        """
        legalMoves = game_state.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Update min_value and min_action for minimizer agent
            if current_value < min_value:
                min_value = current_value
                min_action = action

            # Update beta value for current minimizer
            beta = min(beta, min_value)

            # Pruning: Returns min_value because next possible min_value(s) of minimizer
            # can get worse for alpha value of maximizer when coming back up
            if min_value < alpha:
                return min_action, min_value

        return min_action, min_value

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.get_value(gameState, 0, 0)


#aleg cea mai buna actiune folosind alg expectimax
        return action

    def get_value(self, game_state, index, depth):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Expectation-agent
        """
        # Terminal states:
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(game_state)

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(game_state, index, depth)

        # Expectation-agent: Ghost has index > 0
        else:
            return self.expected_value(game_state, index, depth)

    def max_value(self, game_state, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = game_state.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_action, current_value = self.get_value(successor, successor_index, successor_depth)

            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_action, max_value

    def expected_value(self, game_state, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = game_state.getLegalActions(index)
        expected_value = 0
        expected_action = ""

        # Find the current successor's probability using a uniform distribution
        successor_probability = 1.0 / len(legalMoves)

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value = self.get_value(successor, successor_index, successor_depth)

            # Update expected_value with the current_value and successor_probability
            expected_value += successor_probability * current_value

        return expected_action, expected_value
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
     
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
# calculam un scor care e bazat pe mai multi factori, care include distanta pana la cea mai apropiata mancare
    #numarul de mancaruri ramase, scorul  curent si nr de capsule


    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)


    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

 
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]
    if food_count > 0:
        closest_food = min(food_distances)


#gaseste distanta de la pacman la ghost
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)
        if ghost_distance < 2:
            closest_food = 99999


#si o sa prrioritizam evitarea fantomelor cand sunt prea aproape
    features = [1.0 / closest_food,
                game_score,
                food_count,
                capsule_count]

    weights = [10,
               200,
               -100,
               -10]
    return sum([feature * weight for feature, weight in zip(features, weights)])
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
