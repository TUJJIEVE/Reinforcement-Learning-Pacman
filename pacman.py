"""
    Program Author -> THUMATI UJJIEVE ( CS16BTECH11039 )
    ** THE CODE IS NICELY COMMENTED, PLEASE LOOK AT THE COMMENTS
    
    **** NOTE ***
        PROGRAM USES ARCADE LIBRARY FOR RENDERING THE ENVIRONMENT. THE ENVIRONMENT IS RENDERED ONLY WHEN RUNNING THE SIMULATION.
    
    Install arcade using pip3 install arcade and install dataclasses using pip3 install dataclasses
    RUN USING " python3 pacman.py "


    REPORT:
        ON EXECUTING THE CODE Q LEARNING WILL BE RUN BY DEFAULT AND THEN A SIMULATION WILL BE DISPLAYED ON THE WINDOW AFTER LEARNING THE 
        ENVIRONMENT.

        The environment is assumed to be 100 X 100 grid. The walls are only present in the boundary. There is only one ghost and one pacman.
        The pacman is depicted by blue color , ghost by red color ,wall by white color and coins by yellow color while simulating.
        The ghost is moved intelligently (always moving closer to the pacman) for only first 10 to 20 moves . After this the ghost moves randomly
        in the grid. This is just to make the pacman learn to avoid the ghost.

        The rewards for picking each coin is 100. For colliding with ghost is -100 and for colliding with boundary is -50. 

        The State is represented by tuple with tuple index

            0 -> pacman row, 1 -> pacman col , 2 -> ghost row, 3-> ghost col , 4 -> pellet_grid represented as string, 5 -> tot_pel


        The learning is done for 100 to 200 episodes and each episode lasting for max of 1000 steps or when terminal state is reached.
        
        OBS:
            Q LEARNING PERFORMED BETTER ON AVERAGE SINCE THE TOTAL REWARDS THE PACMAN WAS COLLECTING ON AVERAGE IN THE LAST FEW EPISODES
            WAS GREATER THAN THAT BY THE SARASA LEARNING ALGORITHM..

        **** NOTE ***:
            Due to memory constraints lesser number of episodes are chosen hence the simulation may be not perfect after learning.
            BUT IT WAS FOUND THAT THE PACMAN WAS ABLE TO COLLECT REWARDS AND AVOID THE GHOST.


"""
import numpy as np
import random
import arcade


""" 
    Rendering Parameters

"""
GAME_SCREEN_LENGTH = 100
GAME_SCREEN_WIDTH = 100
SPRITE_SCALING_PLAYER = 0.3
SPRITE_SCALING_GHOST = 0.3
SPRITE_SCALING_COIN = 0.3
SPRITE_SCALING_WALL = 0.3
class Renderer(arcade.Window):
    """
        Class that defines the Renderer, It takes in widht and height of the window to establish, Uses arcade library
        To install arcade library use pip3 install arcade and pip3 install dataclasses

    """
    def __init__(self,width,height):
        super().__init__(width,height)
        arcade.set_background_color(arcade.color.AMAZON)
        self.learner = None
        self.isterminal = False

    def setup(self,curr_state,grid):
        """
            Method to setup the renderer , This method sets the sprites of the wall, ghost and pacman. Also sets the position of the 
            player, ghost and the pellet grid given the state of the environment.

        """
        self.isterminal = False
        self.player_list = arcade.SpriteList()
        self.ghost_list = arcade.SpriteList()
        self.coin_list = arcade.SpriteList()
        self.wall_list = arcade.SpriteList()
        self.score = 0
        self.curr_state = curr_state
        self.pac_sprite = arcade.Sprite("pac.png",SPRITE_SCALING_PLAYER)
        self.ghost_sprite = arcade.Sprite("ghost.png",SPRITE_SCALING_GHOST)
        self.pac_sprite.center_x = curr_state[1]
        self.pac_sprite.center_y = curr_state[0]
        self.ghost_sprite.center_x = curr_state[3]
        self.ghost_sprite.center_y = curr_state[2]
        self.ghost_list.append(item = self.ghost_sprite)
        self.player_list.append(item = self.pac_sprite)
        pellet_grid = np.fromstring(curr_state[4],dtype = int).reshape(GAME_SCREEN_LENGTH,GAME_SCREEN_WIDTH)
        
        ## Loop to setup the coins on the window 
        for i in range(GAME_SCREEN_LENGTH):
            for j in range(GAME_SCREEN_WIDTH):
                if pellet_grid[i,j] == 1:
                    coin_sprite = arcade.Sprite("coin.jpg",SPRITE_SCALING_COIN)
                    coin_sprite.center_x = j
                    coin_sprite.center_y = i
                    (self.coin_list).append(item = coin_sprite)
        
        ## Loop to setup the walls on the window
        for i in range(GAME_SCREEN_LENGTH):
            for j in range(GAME_SCREEN_WIDTH):
                if grid[i,j] == 0:
                    wall_sprite = arcade.Sprite("wall.jpeg",SPRITE_SCALING_WALL)
                    wall_sprite.center_x = j
                    wall_sprite.center_y = i
                    (self.wall_list).append(item = wall_sprite)
    
    def update_explicit(self,npr,npc,ngr,ngc,pellet_grid,nullcond = False):
        """
            Method to explicitly update the position of the pacman and ghost

        """
        self.pac_sprite.center_x = npc
        self.pac_sprite.center_y = npr
        self.ghost_sprite.center_x = ngc
        self.ghost_sprite.center_y = ngr
        self.coin_list.update()

    def update_coins(self,nullcond,pellet_grid):
        """
            Method to update the coins placed on the grid. This method replaces the coins on the grid once all the coins have been
            exhausted

        """
        if nullcond == True:
            for i in range(GAME_SCREEN_LENGTH):
                for j in range(GAME_SCREEN_WIDTH):
                    if pellet_grid[i,j] == 1:
                        coin_sprite = arcade.Sprite("coin.jpg",SPRITE_SCALING_COIN)
                        coin_sprite.center_x = j
                        coin_sprite.center_y = i
                        (self.coin_list).append(item = coin_sprite)
        
    
    def checkForCollision(self):
        """
            Method to check for collision between the objects that are present in the environment. It returns a tuple that contains 
            boolean values to check wether the player collided with wall , wether the player collided with ghost and wether the player
            collided with the coin

        """
        ### checking for collision between the pacman and the coins
        coins_hit_list = arcade.check_for_collision_with_list(self.pac_sprite,self.coin_list)
        coin_cond = False
        
        if len(coins_hit_list) > 0 :
            coin_cond = True
        ## removing the coins from the rendered environment 
        for coin in coins_hit_list:
            coin.remove_from_sprite_lists()

        ## Checkinf for collision between the pacman and the wall                  
        boundary_hit_list = arcade.check_for_collision_with_list(self.pac_sprite,self.wall_list)
        boundary_cond = False
        if len(boundary_hit_list) > 0:
            boundary_cond = True

        ### Checking for collision between the ghost and the pacman
        ghost_hit_list = arcade.check_for_collision_with_list(self.pac_sprite,self.ghost_list)
        ghost_cond = False
        if len(ghost_hit_list) > 0 :
            ghost_cond = True

        return coin_cond,coins_hit_list,boundary_cond,ghost_cond

    def render(self):
        """
            Method to render the environment
        """
        arcade.run()
    def on_draw(self):
        """
            Method which is continuosly called in event loop once the rendering is started
        """
        # print("Drawing")
        if self.isterminal == True:
            arcade.finish_render()
            print("GAME OVER , TERMINAL STATE REACHED")
            return 0
        arcade.start_render()
        self.wall_list.draw()
        self.coin_list.draw()
        self.ghost_list.draw()
        self.player_list.draw()
        ## Simulate each step of the game
        self.isterminal = self.learner.simulate()

    def on_key_press(self,key,modifiers):
        """
            Method to close the window on key press
        """
        if key == arcade.key.Q:
            arcade.close_window()          
        #print("over")
    def update(self,delta_time):
        pass

class Environment:
    """
        Class which defines the structure of the environment. The environment is of vanilla version of pacman.
    """
    def __init__(self,rows,col,numOfPellets,renderer):
        self.r = rows
        self.c = col
        self.numActions = 5
        self.numPellets = numOfPellets
        self.renderer = renderer
        ## Setting the rewards
        self.pellet_reward = 100
        self.ghost_reward = -100
        self.bound_reward = -50 ## reward when pacman collides with the wall
        self.acc_reward = 0
        self.setupGrid()
        
    def setupPellets(self):
        """
            Method which sets the pellets on the grid. It randomly places the given number of pellets on the grid.
        """
        self.pellet_grid = np.zeros(shape = (self.r,self.c),dtype= int)
        rr =[]
        cc =[]
        for i in range(self.numPellets):
            rrr = random.randint(0,self.r-1)
            ccc = random.randint(0,self.c-1)
        
            while rrr in rr or ccc in cc:
                rrr = random.randint(0,self.r-1)
                ccc = random.randint(0,self.c-1)
                
            rr.append(random.randint(0,self.r-1))
            cc.append(random.randint(0,self.c-1))
        for i in range(self.numPellets):
            self.pellet_grid[rr[i],cc[i]] = 1
        self.pellet_grid = np.pad(self.pellet_grid, ((1,1),(1,1)), 'constant')
        
    def setupGrid(self,givenstate = None):
        """
            Method which sets the grid . It randomly places the pacman in the center area. It also randomly places the ghost
            and sets up the pellets on the grid. If a givenstate is not None then the environment is set according to the user choice which is
            described in the givenstate variable.
        """
        self.iters = 0
        ## if user choice is not present then randomly setup
        if givenstate == None:
            self.setupPellets()
            self.pac_r = random.randint(35,GAME_SCREEN_LENGTH - 35)
            self.pac_c = random.randint(35,GAME_SCREEN_WIDTH - 35)
            self.gh_c = random.randint(18,28)
            while self.gh_c == self.pac_c:
                self.gh_c = random.randint(18,28)
            self.gh_r = self.pac_r
        ## if user choice is present
        else:
            self.pac_r = givenstate[0]
            self.pac_c = givenstate[1]
            self.gh_r = givenstate[2]
            self.gh_c = givenstate[3]
            self.pellet_grid = np.fromstring(givenstate[4],dtype = int).reshape(self.r+2,self.c+2)

        self.grid = np.ones((self.r,self.c),dtype = int)
        self.grid = np.pad(self.grid,((1,1),(1,1)),'constant')
        self.init_state = tuple((self.pac_r,self.pac_c,self.gh_r,self.gh_c,self.pellet_grid.tostring(),self.numPellets))#,self.acc_reward))
        self.curr_state = self.init_state
        
    ## Method to move the elements in the grid
    def isSafeToMove(self,r,c):
        if r>=0 and r <= self.r+1 and c>=0 and c <=self.c + 1:
            return True
        else:
            return False

    def makeMove(self,acPac,curr_state):
        return self.getNextState(curr_state,acPac)
        

    ''' Index of the actions 0 -> right , 1 -> down , 2 -> left , 3 -> up , 4 -> stay '''
    def makePacManMove(self,curr_state,pelletGrid,acPac):
        """
            Method to make the pacman move on the grid. It takes the current position of the pacman and the action number 
        """
        #isTerminal = False
        dirr = [[0,1],[1,0],[0,-1],[-1,0],[0,0]]
        new_r = curr_state[0] + dirr[acPac][0]
        new_c = curr_state[1] + dirr[acPac][1]
        return new_r,new_c
 
    def makeGhostMove(self,curr_state,pacmanRow,pacmanCol):
        """
            This method makes the ghost move around the environment. For about 10 moves the ghost moves inteligently moving closer and closer
            towards the pacman. After 10 moves it randomly moves around the grid.
        """
        dirr = [[0,1],[1,0],[0,-1],[-1,0],[0,0]]
        ## If less than 10 moves happen then intelligently move
        if self.iters <10:
            olg_r = curr_state[2]
            olg_c = curr_state[3]
            dir1 = pacmanRow - olg_r
            dir2 = pacmanCol - olg_c
            select_r = -1
            select_c = -1
            if dir1 > 0 :
                select_r = random.randint(0,1)
            else:
                select_r = random.randint(-1,0)
            if dir2 > 0 :
                select_c = random.randint(0,1)
            else:
                select_c = random.randint(-1,0)
            if [select_r,select_c] in [[1,1],[-1,1],[1,-1],[-1,-1]]:
                select_c = 0
                select_r = 0
        ## if more than 10 moves happen
        else:
            move = random.randint(0,len(dirr)-1)
            select_r = dirr[move][0]
            select_c = dirr[move][1]
        
        newg_r = curr_state[2] + select_r
        newg_c = curr_state[3] + select_c

        """
        Condition to check wether the ghost collided with wall or not. If ghost gets collided with the wall then reset the ghost
        Position according to the rules
        """ 
        if self.grid[newg_r,newg_c] == 0:
            if random.randint(0,1) == 0:
                newg_r = pacmanRow
                if random.randint(0,1) == 0:
                    newg_c = 1
                else:
                     newg_c = self.c
            else :
                newg_c = pacmanCol
                if random.randint(0,1) == 0:
                    newg_r = 1
                else:
                     newg_r = self.r
        return newg_r,newg_c
    
    """ 
        state tuple indexing 
        0 -> pacman row, 1 -> pacman col , 2 -> ghost row, 3-> ghost col , 4 -> pellet_grid, 5 -> tot_pel, 6-> accmulated_reward 
    
    """
    def getNextState(self,curr_state,acPac):
        """
            Method to get the next state of the evironment given the current state and the action to be performed by the pacman.
            The method first checks if the pellets are exhausted or not

            The method makes the pacman move first then makes the ghost move accordingly.
            It then checks for collisions between the objects and updates the environment accordingly.
        
        """
        ## checking if the pellets are exhausted or not
        old_pellet_grid = np.fromstring(curr_state[4],dtype = int).reshape(self.r+2,self.c+2)
        pelletGrid = old_pellet_grid
        nullcond = False
        isTerminal = False
        r,_ = np.where(old_pellet_grid == 1)
        if len(r) == 0:
            self.setupPellets()
            pelletGrid = self.pellet_grid
            nullcond = True        
            self.renderer.update_coins(nullcond,pelletGrid)

        ## making the pacman move        
        npr,npc= self.makePacManMove(curr_state,pelletGrid,acPac) 
        ## making the ghost move
        ngr,ngc = self.makeGhostMove(curr_state,npr,npc)
        ### Update the rendered environment
        self.renderer.update_explicit(npr,npc,ngr,ngc,pelletGrid,nullcond= nullcond)
        ## check for collisions
        coin_cond,coin_list,bound_cond,ghost_cond = self.renderer.checkForCollision()

        new_pellet_grid = np.array(pelletGrid)
        pel_remain = curr_state[5]
        reward = 0
        ## if collided with coin then get the reward
        if coin_cond == True:
            for coins in coin_list:
                new_pellet_grid[coins.center_y,coins.center_x] = 0
                pel_remain -=1
                reward += self.pellet_reward                        
        ## if collided with ghost or boundary then terminal state is reached and collect the reward
        if bound_cond or ghost_cond:
            if ghost_cond:
                # print("##Ghost collided")
                reward += self.ghost_reward
            if bound_cond:
                # print("##Boundary breached")
                reward +=self.bound_reward
            isTerminal = True            
        ### Return the next_state that is achieved based on the current move
        new_state = tuple((npr,npc,ngr,ngc,new_pellet_grid.tostring(),pel_remain))#,acc_rew))
        return new_state,reward,isTerminal

class QSATable:
    """
        Class for the QSA table. This class lays down the structure of the table and necessary methods to get and set the values 
        in the table
    """
    def __init__(self,init_state,num_actions,init_val):
        self.Q_sa = {}
        self.init_val = init_val
        for i in range(num_actions):
            self.Q_sa[(init_state,i)] = init_val
    def getVal(self,state,action):
        """
            Method to get the value from the table given state and action
        """
        if self.isPresent(state,action):
            return self.Q_sa[(state,action)]
        else:
            ## if the tuple entry is not there then return the default value
            return self.init_val
    
    def isPresent(self,state,action):
        """
            Method to check wether the table contains the key (state,action) tuple
        """
        if (state,action) in self.Q_sa:
            return True
        else :
            return False

    def setVal(self,state,action,val):
        """
            Method to set the value given the action,state and the value
        """
        self.Q_sa[(state,action)] = val

class Learner:
    """
        Class for the Learner, it requires hyperparameters value and the environment.
    """
    def __init__(self,gamma,alpha,epsilon,env,init_val,numEpisodes):
        
        self.gamma = gamma
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = numEpisodes
        self.Q_sa = QSATable(self.env.init_state,self.env.numActions,init_val)


    def epsilon_greedy(self,state):
        """
            Method that returns the action number following the epsilon greedy policy. More the value of epsilon more the exploration

        """
        if random.uniform(0,1) > epsilon:
            maxi = -1
            actions = []
            for i in range(self.env.numActions):
                if maxi <= self.Q_sa.getVal(state,i):
                    if maxi == self.Q_sa.getVal(state,i):
                       actions.append(i) 
                    else:
                        maxi = self.Q_sa.getVal(state,i)
                        actions = [i]

            return actions[random.randint(0,len(actions)-1)]
        else:
            return random.randint(0,self.env.numActions-1)
    def QLearner(self):
        """
        Method which implements the Qlearn algorithm based on the given environment. It runs for given number of episodes and each
        episode lasting for 1000 steps

        """
        max_steps = 1000
        numEpisodes = self.episodes
        REW = 0
        for eps in range(numEpisodes):
            self.env.setupGrid()
            self.env.renderer.setup(self.env.curr_state,self.env.grid)
            total_reward = 0
            for steps in range(max_steps):
                #print("step:",steps)
                self.env.iters +=1
                action = self.epsilon_greedy(self.env.curr_state)
                next_state,reward,isTerminal = self.env.getNextState(self.env.curr_state,action)
                total_reward += reward
                maxi = 0
                action_s = []
                for i in range(self.env.numActions):
                    if maxi <= self.Q_sa.getVal(next_state,i):
                        if maxi == self.Q_sa.getVal(next_state,i):
                            action_s.append(i)
                        else:
                            maxi = self.Q_sa.getVal(next_state,i)
                            action_s = [i]
                action_ = random.randint(0,len(action_s)-1)
                if isTerminal:
                    self.Q_sa.setVal(self.env.curr_state,action,self.Q_sa.getVal(self.env.curr_state,action) + self.alpha * (reward - self.Q_sa.getVal(self.env.curr_state,action)) )
                else:
                    self.Q_sa.setVal(self.env.curr_state,action,self.Q_sa.getVal(self.env.curr_state,action) + self.alpha * (reward + ( self.gamma * self.Q_sa.getVal(next_state,action_)) - self.Q_sa.getVal(self.env.curr_state,action)))

                self.env.curr_state = next_state

                if isTerminal:
                    break
            print("Eps:",eps,"Reward in the episode:",total_reward)
            if eps > (numEpisodes*3)//4:
                REW +=total_reward
        return REW/(numEpisodes/4)
    def SARSALearner(self):
        """
            Method which implements SARSA Learning algorithm. It runs for given number of episodes and each episode running for
            1000 steps
        """
        max_steps = 1000
        numEpisodes = self.episodes
        REW = 0
        for eps in range(numEpisodes):
            self.env.setupGrid()
            self.env.renderer.setup(self.env.curr_state,self.env.grid)
            action = self.epsilon_greedy(self.env.curr_state)
            total_reward = 0
            for steps in range(max_steps):
                #print(action)
                self.env.iters+=1
                next_state,reward,isTerminal = self.env.getNextState(self.env.curr_state,action)
                total_reward += reward
                action_ = self.epsilon_greedy(next_state)
                if isTerminal:
                    self.Q_sa.setVal(self.env.curr_state,action,self.Q_sa.getVal(self.env.curr_state,action) + self.alpha * (reward - self.Q_sa.getVal(self.env.curr_state,action)) )
                else:
                    self.Q_sa.setVal(self.env.curr_state,action,self.Q_sa.getVal(self.env.curr_state,action) + self.alpha * (reward + ( self.gamma * self.Q_sa.getVal(next_state,action_)) - self.Q_sa.getVal(self.env.curr_state,action)))
                #print("reward:",reward)
                self.env.curr_state, action = next_state , action_

                if isTerminal:
                    #print("Terminal state Reached:" ,total_reward)
                    break
            print("Eps:",eps,"Reward in the episode:",total_reward)
            if eps > (numEpisodes*3)//4:
                REW +=total_reward
        return REW/(numEpisodes/4)
    def simulate(self):
        """
            Method to simulate the game. The learned agent follows greedy policy and chooses the action that maximizes the value
            in the Q_sa table given the current state.
        """
        print("Playing")
        self.env.iters+=1
        isterminal = False
        reward = 0
        maxi = 0
        actions = []
        for i in range(self.env.numActions):
            if maxi <= self.Q_sa.getVal(self.env.curr_state,i):
                if maxi == self.Q_sa.getVal(self.env.curr_state,i):
                    actions.append(i) 
                else:
                    maxi = self.Q_sa.getVal(self.env.curr_state,i)
                    actions = [i]

        action = actions[random.randint(0,len(actions)-1)]
        #print(action,len(actions))
        next_state,curr_reward,isterminal = self.env.getNextState(self.env.curr_state,action)
        self.env.curr_state = next_state
        reward +=curr_reward
        #print("Curr reward:",reward)    
        return isterminal

    def test(self):
        """
            Method to test the learned agent on the given environment. Method is executed after learning.
            The testing is done by choosing greedy policy 
        """
        self.env.setupGrid(self.env.init_state)
        self.env.renderer.setup(self.env.curr_state,self.env.grid)
        arcade.run()
    


''' hyperparameters for the algorithm '''

gamma = 0.999
alpha = 0.6
epsilon = 0.4
init_val = 0.3
''' Change to increase the number of episodes '''
episodes = 100
## renderer object
renderer = Renderer(GAME_SCREEN_WIDTH,GAME_SCREEN_LENGTH)
## environment object
env = Environment(GAME_SCREEN_LENGTH-2,GAME_SCREEN_WIDTH-2,4,renderer)



""" Q LEARNING PART """

Qlearner = Learner(gamma,alpha,epsilon,env,init_val,episodes)
print("Learning WITH Q")
qrew = Qlearner.QLearner()
print("Learnt with Q storing reward")
print("ON AVG. REWARD GAINED IN Q learning COMPUTED FOR LAST",episodes//4,"EPISODES",qrew)
print ("To know more about the SARSA algorithm... Run in another program after uncommenting in the code")
print ("Follow the steps to not hang the system and to reduce the usage of RAM")
''' comment to not to run the testing of Q'''
renderer.learner = Qlearner
Qlearner.test()

""" SARSA PART """

''' UNCOMMENT TO RUN SARSA and to get the avg reward collected'''
# SARSAlearner = Learner(gamma,alpha,epsilon,env,init_val,episodes)
# print("Learning WITH SARSA")
# sarsarew = SARSAlearner.SARSALearner()
# print("Learnt with SARSA storing reward")
# print("ON AVG. REWARD GAINED IN SARSA COMPUTED FOR LAST",episodes//4,"OF EPISODES",sarsarew)
''' Uncomment to run the testing of SARSA'''
# renderer.learner = Qlearner
# Qlearner.test()