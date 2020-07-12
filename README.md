<h1 align="center"> MusicBox </h1>
<p align="center">
  <img width="460" height="300" src=other/music.gif>
</p>


## Music just for you

MusicBox is my project as an AI fellow at Insight Data Science. The goal of the project is to build a new generation of recommender system that learns patterns of user behavior through deep reinforcement learning.

## Overview

To build MusicBox with the capability to handle a large number of discrete actions (e.g. millions of songs in database), I implemented the model Deep Deterministic Policy Gradient (DDPG) with Tensorflow.

DDPG is an algorithm that can concurrently learn a policy and a Q-function. DDPG adopts an Actor-Critic scheme to model the sequential interactions between the users and recommender system. Based on the songs that a user has been listening to in a listening session, actor network takes the embedding of song features, builds a policy function that scores all music, and then recommends top three songs with the highest scores for the user. Next, Critic Network uses approximation to learn a Q-value function, which judges if selected songs match the current state of user behavior. According to the judgement from the Critic network, the Actor network updates its’ policy parameters to improve recommending performance in the following iterations.

<p align="center">
  <img width="400" height="330" src=other/ddpg.png>
</p>

## Evaluation

To measure MusicBox's performance offline, I tracked multiple metrics such as music diversity and song skip rates. Compared to the original methods used in the dataset, MusicBox recommends a more diverse selection of music (9.3 % vs 7.6%) and also better identifies songs that are skipped by users (49% vs 54%).

## Limitations

One major impediment of applying reinforcement learning to recommender system is the lack of simulation platforms for sequential user interactions. This makes the full evaluation of MusicBox difficult, especially when it comes to reasoning about ordering of songs in a playlist.

## Example usage

```
cd src
from main import *
run('../data/user_mini_data.tar.gz', '../data/music_mini_data.tar.gz')
```

## Data

I built MusicBox based on a real-world dataset from Spotify. As entire dataset is very large (e.g. 130 million listening sessions), I will not store it here on github. If you are interested, please check out their website [https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge).

## Inspiration

1 [https://arxiv.org/abs/1801.00209](https://arxiv.org/abs/1801.00209)

2 [https://github.com/egipcy/LIRD](https://github.com/egipcy/LIRD)

3 [https://github.com/luozachary/drl-rec](https://github.com/luozachary/drl-rec)
