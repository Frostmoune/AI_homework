# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
import time
import heapq
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def getRoad(pre_states, now_state, res, num, directions):
    try:
        next_states = {}
        next_states[now_state] = None
        while pre_states[now_state] != None:
            next_states[pre_states[now_state]] = now_state
            now_state = pre_states[now_state]
        
        res_state = now_state

        now_state = next_states[now_state]
        while now_state != None:
            res.append(now_state[1])
            now_state = next_states[now_state]
        
        return res_state
    except Exception as e:
        print(e)

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    ##  run:      python pacman.py -l tinyMaze -p SearchAgent
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    start = problem.getStartState()
    def myDFS(location, res, direction):

        if problem.isGoalState(location):
            return True
        successors = problem.getSuccessors(location)
        for nextLoc in successors:
            pos = nextLoc[0]
            # val = pos[0] * 10 + pos[1]
            if pos in marked:
                continue
            else:
                marked[pos] = 1
            res.append(nextLoc[1])
            if myDFS(pos, res, nextLoc[1]):
                return True

        if direction != None:
            if direction == s:
                res.append(n)
            elif direction == n:
                res.append(s)
            elif direction == e:
                res.append(w)
            else:
                res.append(e)
        return False

    res = []
    marked = {}
    marked[start[0] * 10 + start[1]] = 1
    if myDFS(start, res, None):
   		print("reach goal")
    print(res)
    return res
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST
    # n = Directions.NORTH
    # e = Directions.EAST

    start = problem.getStartState()
    marked = {}
    pre_states = {}
    pre_states[start] = None
    marked[start[0] * 10 + start[1]] = 1
    end_state = None

    search_queue = util.Queue()
    search_queue.push(start)

    while not search_queue.isEmpty():
        now_state = search_queue.pop()
        if now_state == start:
            now_location = start
        else:
            now_location = now_state[0]
        if problem.isGoalState(now_location):
            end_state = now_state
            break
        successors = problem.getSuccessors(now_location)
        for nextLoc in successors:
            pos = nextLoc[0]
            if pos in marked:
                continue
            else:
                marked[pos] = 1
            search_queue.push(nextLoc)
            pre_states[nextLoc] = now_state 

    res = []
    if end_state != None:
        # getRoad(pre_states, end_state, res)
        print("reach goal")
    print(res)
    return res
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def clearQueue(search_queue):
    while not search_queue.isEmpty():
        search_queue.pop()

def decoding(now_hash, directions, top, right):
    direction = directions[now_hash % 4]
    now_hash //= 4
    col = now_hash % top
    row = now_hash // top
    return ((row + 1, col + 1), direction)

def encoding(state, directions, top, right):
    if not isinstance(state[0], int):
        if isinstance(state[0][0], int):
            now_hash = (state[0][0] - 1) * top + (state[0][1] - 1)
        else:
            now_hash = (state[0][0][0] - 1) * top + (state[0][0][1] - 1)
        now_hash *= 4
        for i in range(4):
            if state[1] == directions[i]:
                now_hash += i
                break
    else:
        now_hash = (state[0] - 1) * top + (state[1] - 1)
    return now_hash

def getPos(state, problem):
    from searchAgents import FoodSearchProblem
    if isinstance(problem, FoodSearchProblem):
        return state[0][0]
    return state[0]

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    def aStarPerSearch(problem, start):
        from searchAgents import FoodSearchProblem, CornersProblem

        marked = {}
        marked[getPos(start, problem)] = 1

        end_state = None

        search_queue = util.PriorityQueue()

        pre_states = []
        search_queue.push((start, []), 0)
        target_num = 0
        pre_states.append({})
        pre_states[target_num][start] = None

        while not search_queue.isEmpty():
            now_state, now_res = search_queue.pop()

            now_is_goal = int(problem.isGoalState(now_state[0]))
            if now_is_goal >= 0:
                if now_is_goal == 1:
                    end_state = now_state
                    break
                if isinstance(problem, CornersProblem) or isinstance(problem, FoodSearchProblem):
                    marked = {}
                    pre_states.append({})
                    target_num += 1
                    clearQueue(search_queue)
                    sub_res = copy.deepcopy(list(now_res))
                    search_queue.push((now_state, now_res), 0)
                    pre_states[target_num][now_state] = None
                    marked[getPos(now_state, problem)] = 1
            
            successors = problem.getSuccessors(now_state[0])
            for next_state in successors:
                pos = getPos(next_state, problem)
                sub_res = copy.deepcopy(list(now_res))
                sub_res.append(next_state[1])
                next_cost = problem.getCostOfActions(sub_res)
                next_heuristic = heuristic(next_state[0], problem)

                try:
                    if pos in marked:
                        for index, (p, c, i) in enumerate(search_queue.heap):
                            if i[0][0] == pos:
                                if p <= next_cost + next_heuristic:
                                    break
                                del search_queue.heap[index]
                                search_queue.heap.append((next_cost + next_heuristic, c, (next_state, sub_res)))
                                heapq.heapify(search_queue.heap)
                                pre_states[target_num][next_state] = now_state
                                break 
                        continue
                    else:
                        marked[pos] = 1
                    search_queue.push((next_state, sub_res), next_cost + next_heuristic)
                    pre_states[target_num][next_state] = now_state
                except Exception as e:
                    print(e)

        return pre_states, end_state, target_num

    start = problem.getStartState()
    pre_states, end_state, target_num = aStarPerSearch(problem, start)
    total_res = []
    res = [[] for _ in range(target_num + 1)]
    if end_state != None:
        for i in range(target_num + 1):
            end_state = getRoad(pre_states[target_num - i], end_state, res[target_num - i], target_num - i, directions)
        for i in range(target_num + 1):
            if i == 0: 
                total_res = res[i]
            else:
                total_res += res[i]
        print("reach goal")

    print(total_res)
    return total_res
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
