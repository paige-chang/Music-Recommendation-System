## Helper functions

import random
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataGenerator():
    
    def __init__(self, datapath, itempath):
        
        '''
        Load data 
        List the users and the music items
        List all the users historic
        '''
        
        self.data  = self.load_datas(datapath, itempath)
        self.users = self.data['session_id'].unique()  #list of all users
        self.histo = self.gen_histo()
        self.train = []
        self.test  = []

    def load_music_data(self, itempath):
        
        '''
        Parameters
        ----------
        datapath :  string
                    path to the music data
        '''

        music = pd.read_csv(itempath, compression = 'gzip')
        music = music.rename(columns = {music.columns[0]:'track_id'})
        music = music[music.us_popularity_estimate>=95] # Focus on the song tracks that have good popularity

        return music	
    
    def load_user_data(self, datapath):
        
        '''
        Parameters
        ----------
        datapath :  string
                    path to the session data
        '''
        
        user = pd.read_csv(datapath, compression = 'gzip', nrows = 100000)
        user = user.rename(columns = {user.columns[0]:'session_id'})

        # Correct the labeling of skip
        # rating = 1 if user doesn't the skip the music, otherwise rating = 0
        user['rating'] = user.skip_2.astype(int)*-1+1
        
        return user
   
    def load_datas(self, datapath, itempath):
        
        '''
        Load the data and merge the id of each song track. 
        A row corresponds to a rate given by a user to a song.
        '''
        
        music = self.load_music_data(itempath)
        data = self.load_user_data(datapath)
        data = pd.merge(data, music, how = 'inner', left_on = ['track_id_clean'], right_on = ['track_id'])
        data = data.groupby(['session_id','session_length']).filter(lambda x: len(x) == 20) # Filter out the sessions that are too short
        data['session_id'] = LabelEncoder().fit_transform(data['session_id'])
        data['music_id'] = LabelEncoder().fit_transform(data['track_id'])
        
        return data
    
    def gen_histo(self):
        
        '''
        Group all rates given by users and store them from older to most recent.
        
        Returns
        -------
        result :    List(DataFrame)
                    List of the historic for each user
        '''
        
        historic_users = []
        for i, u in enumerate(self.users):
            temp = self.data[self.data['session_id'] == u]
            temp = temp.sort_values('session_position').reset_index()
            temp.drop('index', axis = 1, inplace = True)
            historic_users.append(temp)
        return historic_users
    
    def sample_histo(self, user_histo, action_ratio = 0.8, max_samp_by_user = 10,  max_state = 10, max_action = 5,
                 nb_states = [], nb_actions = []):
        
        '''
        For a given historic, make one or multiple sampling.
        If no optional argument given for nb_states and nb_actions, then the sampling
        is random and each sample can have differents size for action and state.
        To normalize sampling we need to give list of the numbers of states and actions
        to be sampled.

        Parameters
        ----------
        user_histo :  DataFrame
                          historic of user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which song tracks in history will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of song tracks to take for the 'state' column
        max_action :      int, optional
                          Number max of song tracks to take for the 'action' action
        nb_states :       array(int), optional
                          Numbers of song tracks to be taken for each sample made on user's historic
        nb_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historic
        
        Returns
        -------
        states :         List(String)
                        All the states sampled, format of a sample: itemId&rating
        actions :        List(String)
                        All the actions sampled, format of a sample: itemId&rating
      

        Notes
        -----
        States must be before(timestamp<) the actions.
        If given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals
        '''

        n = len(user_histo)
        sep = int(action_ratio * n)
        nb_sample = random.randint(1, max_samp_by_user)
        if not nb_states:
            nb_states = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
        if not nb_actions:
            nb_actions = [min(random.randint(1, n - sep), max_action) for i in range(nb_sample)]
        assert len(nb_states) == len(nb_actions), 'Given array must have the same size'
        
        states  = []
        actions = []
        # Select samples in histo 
        for i in range(len(nb_states)):
            sample_states = user_histo.iloc[0:sep].sample(nb_states[i])
            sample_actions = user_histo.iloc[-(n - sep):].sample(nb_actions[i])
            
            sample_state  = []
            sample_action = []
            for j in range(nb_states[i]):
                row = sample_states.iloc[j]
                # Formate State
                state = str(row.loc['music_id'])+'&'+str(row.loc['rating'].astype(int))
                sample_state.append(state)
          
            for j in range(nb_actions[i]):
                row = sample_actions.iloc[j]
                # Formate Action 
                action = str(row.loc['music_id'])+'&'+str(row.loc['rating'].astype(int))
                sample_action.append(action)

            states.append(sample_state)
            actions.append(sample_action)
                
        return states, actions
    
    def gen_train_test(self, train_ratio, seed = None):
        
        '''
        Shuffle the historic of users and separate it in a train and a test set.
        Store the ids for each set.
        An user can't be in both set.

        Parameters
        ----------
        test_ratio :  float
                      Ratio to control the sizes of the sets
        seed       :  float
                      Seed on the shuffle
        '''
        
        n = len(self.histo)

        if seed is not None:
            random.Random(seed).shuffle(self.histo)
        else:
            random.shuffle(self.histo)

        self.train = self.histo[:int((train_ratio * n))]
        self.test  = self.histo[int((train_ratio * n)):]
        self.user_train = [h.iloc[0,0] for h in self.train]
        self.user_test  = [h.iloc[0,0] for h in self.test]
        
    def write_csv(self, filename, histo_to_write, delimiter=';', action_ratio = 0.8, max_samp_by_user = 10,
                  max_state = 10, max_action = 5, nb_states = [], nb_actions = []):
        
        '''
        From  a given historic, create a csv file with the format:
        columns : state;action_reward;n_state
        rows    : itemid&rating1 | itemid&rating2 | ... ; itemid&rating3 | ...  | ... | item&rating4
        at filename location.

        Parameters
        ----------
        filename :        string
                          path to the file to be produced
        histo_to_write :  List(DataFrame)
                          List of the historic for each user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which song tracks in history will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of song tracks to take for the 'state' column
        max_action :      int, optional
                          Number max of song tracks to take for the 'action' action
        nb_states :       array(int), optional
                          Numbers of song tracks to be taken for each sample made on user's historic
        nb_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historic

        Notes
        -----
        if given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals

        '''
        
        with open(filename, mode = 'w') as file:
            f_writer = csv.writer(file, delimiter = delimiter)
            f_writer.writerow(['state', 'action_reward', 'n_state'])
            for user_histo in histo_to_write:
                states, actions = self.sample_histo(user_histo, action_ratio, max_samp_by_user, max_state, max_action, nb_states, nb_actions)
                for i in range(len(states)):
                    # FORMAT STATE
                    state_str   = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str  = '|'.join(actions[i])
                    # FORMAT N_STATE
                    n_state_str = state_str + '|' + action_str
                    f_writer.writerow([state_str, action_str, n_state_str])
                    
                    
class Noise():
    
    '''
    Noise for Actor predictions
    '''
    
    def __init__(self, action_space_size, mu = 0, theta = 0.5, sigma = 0.2):
        self.action_space_size = action_space_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_space_size) * self.mu

    def get(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.rand(self.action_space_size)
        return self.state
    
    
def read_file(data_path):
    
    '''
    Load data from train_set.csv or test_set.csv
    '''

    data = pd.read_csv(data_path, sep = ';')
    for col in ['state', 'n_state', 'action_reward']:
        data[col] = [np.array([[np.int(float(k)) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
    for col in ['state', 'n_state']:
        data[col] = [np.array([e[0] for e in l]) for l in data[col]]

    data['action'] = [[e[0] for e in l] for l in data['action_reward']]
    data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
    data.drop(columns = ['action_reward'], inplace = True)

    return data

def read_embeddings(embeddings_path):
    
    ''' Load embeddings (a vector for each item)'''
    
    embeddings = pd.read_csv(embeddings_path, sep = ';')

    return np.array([[np.float64(k) for k in e.split('|')] for e in embeddings['vectors']])

