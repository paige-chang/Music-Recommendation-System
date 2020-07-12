## Model training

from model import *
from utilities import Noise
import tensorflow as tf
import time

def agent_train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary, nb_rounds):

    # Set up summary operators
    def build_summaries():
        episode_reward = tf.Variable(0.)
        tf.summary.scalar('reward', episode_reward)
        episode_max_Q = tf.Variable(0.)
        tf.summary.scalar('max_Q_value', episode_max_Q)
        critic_loss = tf.Variable(0.)
        tf.summary.scalar('critic_loss', critic_loss)

        summary_vars = [episode_reward, episode_max_Q, critic_loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(filename_summary, sess.graph)

    # Initialize target network f′ and Q′'
    actor.init_target_network()
    critic.init_target_network()

    # Initialize the capacity of replay memory D'
    replay_memory = ReplayMemory(buffer_size) # Memory D 
    replay = False

    start_time = time.time()
    for i_session in range(nb_episodes): 
        session_reward = 0
        session_Q_value = 0
        session_critic_loss = 0

        states = environment.reset() # Initialize state s_0 from previous sessions
        
        if (i_session + 1) % 10 == 0: # Update average parameters every 10 episodes
            environment.groups = environment.get_groups()
          
        exploration_noise = Noise(history_length * embeddings.size())

        for t in range(nb_rounds): 
            # Transition Generating Stage
            # Select an action a_t = {a_t^1, ..., a_t^K}
            actions = actor.get_recommendation_list(
                ra_length,
                states.reshape(1, -1) + exploration_noise.get().reshape(1, -1),
                embeddings).reshape(ra_length, embeddings.size())

            # Execute action a_t and observe the reward list {r_t^1, ..., r_t^K} for each item in a_t'
            rewards, next_states = environment.step(actions)

            # 'Store transition (s_t, a_t, r_t, s_t+1) 
            replay_memory.add(states.reshape(history_length * embeddings.size()),
                              actions.reshape(ra_length * embeddings.size()),
                              [rewards],
                              next_states.reshape(history_length * embeddings.size()))

            states = next_states # Set s_t = s_t+1'

            session_reward += rewards
            
            # Parameter Updating Stage'
            if replay_memory.size() >= batch_size: # Experience replay
                replay = True
                replay_Q_value, critic_loss = experience_replay(replay_memory, batch_size,
                  actor, critic, embeddings, ra_length, history_length * embeddings.size(),
                  ra_length * embeddings.size(), discount_factor)
                session_Q_value += replay_Q_value
                session_critic_loss += critic_loss

            summary_str = sess.run(summary_ops,
                                  feed_dict = {summary_vars[0]: session_reward,
                                              summary_vars[1]: session_Q_value,
                                              summary_vars[2]: session_critic_loss})
            
            writer.add_summary(summary_str, i_session)


        str_loss = str('Loss = %0.4f' % session_critic_loss)
        print(('Episode %d/%d Time = %ds ' + (str_loss if replay else 'No replay')) % (i_session + 1, nb_episodes, time.time() - start_time))

        start_time = time.time()
        
    writer.close()
