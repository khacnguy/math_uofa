import heapq
import math
import sys

class State:
    """
    Class to represent a state on grid-based pathfinding problems. The class contains two static variables:
    map_width and map_height containing the width and height of the map. Although these variables are properties
    of the map and not of the state, they are used to compute the hash value of the state, which is used
    in the CLOSED list. 

    Each state has the values of x, y, g, h, and cost. The cost is used as the criterion for sorting the nodes
    in the OPEN list for both Dijkstra's algorithm and A*. For Dijkstra the cost should be the g-value, while
    for A* the cost should be the f-value of the node. 
    """
    map_width = 0
    map_height = 0
    
    def __init__(self, x, y):
        """
        Constructor - requires the values of x and y of the state. All the other variables are
        initialized with the value of 0.
        """
        self._x = x
        self._y = y
        self._g = 0
        self._cost = 0
        self._f = 0
        
    def __repr__(self):
        """
        This method is invoked when we call a print instruction with a state. It will print [x, y],
        where x and y are the coordinates of the state on the map. 
        """
        state_str = "[" + str(self._x) + ", " + str(self._y) + "]"
        return state_str
    
    def __lt__(self, other):
        """
        Less-than operator; used to sort the nodes in the OPEN list
        """
        return self._g< other._g
    
    def state_hash(self):
        """
        Given a state (x, y), this method returns the value of x * map_width + y. This is a perfect 
        hash function for the problem (i.e., no two states will have the same hash value). This function
        is used to implement the CLOSED list of the algorithms. 
        """
        return self._y * State.map_width + self._x
    
    def __eq__(self, other):
        """
        Method that is invoked if we use the operator == for states. It returns True if self and other
        represent the same state; it returns False otherwise. 
        """
        return self._x == other._x and self._y == other._y

    def get_x(self):
        """
        Returns the x coordinate of the state
        """
        return self._x
    
    def get_y(self):
        """
        Returns the y coordinate of the state
        """
        return self._y
    
    def get_g(self):
        """
        Returns the g-value of the state
        """
        return self._g
        
    def set_g(self, g):
        """
        Sets the g-value of the state
        """
        self._g = g

    def get_cost(self):
        """
        Returns the cost of a state; the cost is determined by the search algorithm
        """
        return self._cost
    
    def set_cost(self, cost):
        """
        Sets the cost of the state; the cost is determined by the search algorithm 
        """
        self._cost = cost

def dijkstra_alg(map, start, goal):
    open_list = []
    closed_list = {}
    i = 0
    open_list.append(start) #already has cost, g = 0
    closed_list[start.state_hash()] = start.get_g()
    while (len(open_list)>0):
        cur_state = heapq.heappop(open_list)
        i += 1
        if cur_state == goal: 
            return cur_state.get_g(), i

        # closed_list[cur_state.state_hash()] = cur_state.get_g()
        for neighbor_state in map.successors(cur_state):
            if neighbor_state.state_hash() not in closed_list or neighbor_state.get_g() < closed_list[neighbor_state.state_hash()]: 
                closed_list[neighbor_state.state_hash()] = neighbor_state.get_g()
                heapq.heappush(open_list, neighbor_state)
    return -1, i

def astar_alg(map, start, goal):
    open_list = []
    closed_list = {}

    heapq.heappush(open_list, (start.get_g() + heuristic(start, goal), start)) #already has cost, g = 0
    closed_list[start.state_hash()] = start.get_g()
    i = 0
    while (len(open_list)>0):
        _, cur_state = heapq.heappop(open_list)
        i += 1
        if cur_state == goal: 
            return cur_state.get_g(), i

        # closed_list[cur_state.state_hash()] = cur_state.get_g()
        for neighbor_state in map.successors(cur_state):
            if neighbor_state.state_hash() not in closed_list or neighbor_state.get_g() < closed_list[neighbor_state.state_hash()]: 
                closed_list[neighbor_state.state_hash()] = neighbor_state.get_g()
                heapq.heappush(open_list, (neighbor_state.get_g() + heuristic(neighbor_state, goal), neighbor_state))
    return -1, i
def heuristic(s1, s2):
    dx = abs(s1.get_x() - s2.get_x())
    dy = abs(s1.get_y() - s2.get_y())
    return 1.5 * min(dx, dy) + abs(dx-dy)
    

    