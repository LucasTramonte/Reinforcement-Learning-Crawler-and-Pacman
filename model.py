import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 1
        self.numTrainingGames = 4000
        self.batch_size = 128
        
        self.layers = [
            nn.Parameter(state_dim, 256), 
            nn.Parameter(256, 128),  
            nn.Parameter(128, 64),   
            nn.Parameter(64, action_dim)
        ]
        
        self.set_weights(self.layers)

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_predictions = self.run(states)
        return nn.SquareLoss(Q_predictions,Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"

        linear_output1 = nn.Linear(states, self.parameters[0])
        relu_output1 = nn.ReLU(linear_output1)
        
        linear_output2 = nn.Linear(relu_output1, self.parameters[1])
        relu_output2 = nn.ReLU(linear_output2)
        
        linear_output3 = nn.Linear(relu_output2, self.parameters[2])
        relu_output3 = nn.ReLU(linear_output3)
        
        return nn.Linear(relu_output3, self.parameters[3])
    
    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        grads = nn.gradients(loss, self.parameters)
        for param, grad in zip(self.parameters, grads):
            param.update(grad, -self.learning_rate)
        

