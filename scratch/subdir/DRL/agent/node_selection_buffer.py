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
        self.terminal_memory = np.zeros(self.mem_size,dtype=bool) #terminal memory we need it to store the done flags
    
    def preprocess_observation(self, observation):
        current_state = observation['current_state']
        fl_accuracy = observation['FL_accuracy']

        # Flatten the current_state component
        flattened_current_state = current_state.flatten()

        # Concatenate the flattened current_state and the FL_accuracy
        processed_observation = np.concatenate((flattened_current_state, [fl_accuracy]))

        return processed_observation

    def store_transition(self,state,action,reward,state_,done):
        # print ("im in store_transition")
        index = self.mem_cntr % self.mem_size
        # print("lest see index", index)
        # print ("lets see state of store transition", state)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1 

    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,batch_size) #randomly choose the batch size from the memory

        states = self.preprocess_observation(self.state_memory[batch]) #get the states
        states_ = self.preprocess_observation(self.new_state_memory[batch]) #get the new states
        actions = self.action_memory[batch] #get the actions
        rewards = self.reward_memory[batch] #get the rewards
        dones = self.terminal_memory[batch] #get the done flags

        return states, actions, rewards, states_, dones
