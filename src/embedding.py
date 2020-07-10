## Word2Vec embeddings for song-user features

class Embeddings():

    '''
    Returns Embeddings and embedding metadata after initialization with item_embeddings
    '''

    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings
    
    def size(self):
        return self.item_embeddings.shape[1]
    
    def get_embedding_vector(self):
        return self.item_embeddings
    
    def get_embedding(self, item_index):
        return self.item_embeddings[item_index]

    def embed(self, item_list):
        return np.array([self.get_embedding(item) for item in item_list])
    

class EmbeddingsGenerator():

    '''
    Returns Embeddings after initialization with train_users and data
    '''

    def  __init__(self, train_users, data):
        
        self.train_users = train_users
        self.data = data.sort_values(by=['date'])
        self.session_count = self.data['session_id'].max()+1
        self.track_count = self.data['music_id'].max()+1
        self.session_tracks = {} # list of rated song tracks by each session
        for sessionId in range(self.session_count):
            self.session_tracks[sessionId] = self.data[self.data.session_id == sessionId]['music_id'].tolist()
        self.m = self.model()

    def model(self, hidden_layer_size = 300):
        
        m = Sequential()
        m.add(Dense(hidden_layer_size, input_shape = (1, self.track_count)))
        m.add(Dense(self.track_count, activation = 'softmax'))
        m.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return m
    
    def generate_input(self, session_id):
        
        '''
        Returns a context and a target for the user_id
        context: user's history with one random song removed
        target: id of random removed song
        '''
        
        session_tracks_count = len(self.session_tracks[session_id])
        # picking random song
        random_index = np.random.randint(0, session_tracks_count-1) # -1 avoids taking the last track
        # setting target
        target = np.zeros((1, self.track_count))
        target[0][self.session_tracks[session_id][random_index]] = 1
        # setting context
        context = np.zeros((1, self.track_count))
        context[0][self.session_tracks[session_id][:random_index] + self.session_tracks[session_id][random_index+1:]] = 1
        return context, target

    def train(self, nb_epochs, batch_size=2000):
        
        '''
        Trains the model from train_users's history
        '''
        
        for i in range(nb_epochs):
            print('%d/%d' % (i+1, nb_epochs))
            batch = [self.generate_input(session_id = np.random.choice(self.train_users)) for _ in range(batch_size)]
            X_train = np.array([b[0] for b in batch])
            y_train = np.array([b[1] for b in batch])
            self.m.fit(X_train, y_train, epochs = 1, validation_split = 0.3)

    def test(self, test_users, batch_size = 2000):
        
        '''
        Returns [loss, accuracy] on the test set
        '''
        
        print('test users', len(test_users))
        batch_test = [self.generate_input(session_id = np.random.choice(test_users)) for _ in range(batch_size)]
        X_test = np.array([b[0] for b in batch_test])
        y_test = np.array([b[1] for b in batch_test])
        return self.m.evaluate(X_test, y_test)

    def save_embeddings(self, file_name):
        
        '''
        Generates a csv file containg the vector embedding for each song
        '''
        
        inp = self.m.input                                           # input placeholder
        outputs = [layer.output for layer in self.m.layers]          # all layer outputs
        functor = K.function([inp, K.learning_phase()], outputs )    # evaluation function

        # append embeddings to vectors
        vectors = []
        for music_id in range(self.track_count):
            track = np.zeros((1, 1, self.track_count))
            track[0][0][music_id] = 1
            layer_outs = functor([track])
            vector = [str(v) for v in layer_outs[0][0][0]]
            vector = '|'.join(vector)
            vectors.append([music_id, vector])

        #saves as a csv file
        embeddings = pd.DataFrame(vectors, columns = ['music_id', 'vectors']).astype({'music_id': 'int32'})
        embeddings.to_csv(file_name, sep = ';', index = False)

