import numpy as np 

class ReplayBuffer():
    #max_size is max memory,n_actions number of actions, input_shape is observation shape
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0 # memory counter to keep track
        self.state_memory = np.zeros((self.mem_size,*input_shape)) #state memory
        self.new_state_memory = np.zeros((self.mem_size,*input_shape)) #new state memory
        self.action_memory = np.zeros((self.mem_size,n_actions)) #action memory
        self.reward_memory = np.zeros(self.mem_size) #reward memory
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool) #terminal memory we need it to store the done flags

    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state #store the state
        self.new_state_memory[index] = state_ #store the new state
        self.action_memory[index]= action#store the action taken in that step
        self.reward_memory[index] = reward #store the reward
        self.terminal_memory[index] = done #store the done flag
        self.mem_cntr += 1 #increment the memory counter  

    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,batch_size) #randomly choose the batch size from the memory

        states = self.state_memory[batch] #get the states
        states_ = self.new_state_memory[batch] #get the new states
        actions = self.action_memory[batch] #get the actions
        rewards = self.reward_memory[batch] #get the rewards
        dones = self.terminal_memory[batch] #get the done flags

        return states, actions, rewards, states_, dones