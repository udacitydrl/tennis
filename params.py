class Params():
    """ DDPG config parameters """
    def __init__(self, units_actor=[256, 256], units_critic=[256, 256], lr_actor = 0.0001, lr_critic=0.0001, 
                 gamma=0.99, tau=0.001, buffer_size=100000, batch_size=128, mu=0, theta=0.15, sigma=0.2):
        """
        Initialize parameters
        
        Params
        ======
            units_actor (array): hidden units of the actor
            units_critic (array): hidden units of the critic
            
            lr_actor (float): learning rate of the actor 
            lr_critic (float): learning rate of the critic
            
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size

            gamma (float): discount factor between 0 and 1
            tau (float): tau for soft update of target parameters

            mu (float): mu in OUNoise process
            theta (float): theta in OUNoise process
            signma (float): sigma in OUNoise process
        """
        self.units_actor = units_actor.copy()
        self.units_critic = units_critic.copy()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        self.mu=mu
        self.theta=theta
        self.sigma=sigma