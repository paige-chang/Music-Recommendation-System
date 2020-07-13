from utilities import *
from embedding import *
from environment import *
from model import *
from train import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def recsys(data, embeddings, history_length, ra_length, discount_factor = 0.99, actor_lr = 0.00005, critic_lr = 0.001,
        tau = 0.001, batch_size = 64, nb_rounds = 50, nb_episodes = 5, alpha = 0.2,gamma = 0.9, buffer_size=10000,
        fixed_length = True):
   
    state_space_size = embeddings.size() * history_length
    action_space_size = embeddings.size() * ra_length
    filename_summary = 'summary.txt' 
    
    environment = Environment(data, embeddings, alpha, gamma, fixed_length)
    tf.reset_default_graph() # For multiple consecutive executions
    sess = tf.Session()

    # Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights
    actor = Actor(sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embeddings.size(), tau, actor_lr)
    critic = Critic(sess, state_space_size, action_space_size, history_length, embeddings.size(), tau, critic_lr)
    agent_train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary, nb_rounds)
    
def run(dataset, track_features, history_length = 10, ra_length=3):

    dg = DataGenerator(dataset, track_features)
    dg.gen_train_test(train_ratio = 0.7, seed = 42)
    dg.write_csv('train_set.csv', dg.train, nb_states = [history_length], nb_actions = [ra_length])
    dg.write_csv('test_set.csv', dg.test, nb_states = [history_length], nb_actions = [ra_length])
    data = read_file('train_set.csv')
    
    if False: # Generate embeddings
        eg = EmbeddingsGenerator(dg.user_train, dg.data)
        eg.train(nb_epochs = 200)
        eg.save_embeddings('embeddings.csv')
    
    embeddings = Embeddings(read_embeddings('embeddings.csv'))
    recsys(data, embeddings, history_length, ra_length)
        
if __name__ == '__main__':
    run('../data/user_mini_data.tar.gz', '../data/music_mini_data.tar.gz')

