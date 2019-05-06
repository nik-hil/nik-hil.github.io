---

title:  "Game playing introduction in artificial intelligence"
---


# WEEK-2 Game playing with AI

## Game playing with AI is hard. We look for strategies for turn based games

1. Game we pick is Isolation. Player1 starts with 'O'
1. ![isolation-game](/assets/aind-1/images/isolation.jpeg?raw=true)
1. Build a tree of all possiblities. Find the branches where player1 will loose. 
1. Teach computer to find loosing branches with a version of [MiniMax algorithm](https://classroom.udacity.com/nanodegrees/nd889/parts/6be67fd1-9725-4d14-b36e-ae2b5b20804c/modules/f719d723-7ee0-472c-80c1-663f02de94f3/lessons/9b1a742a-fa2d-4940-922c-ed426b44f81b/concepts/49461227440923).

## Max number of nodes visited.

1. for 5 x 5 board of isolation game, at each turn number of available branches are , `25 * 24 * 23 * ... * 3 * 2 * 1 = 10^25`. Not feasible.

## Branching factor
1. Consider branching factor. After each player turn no of empty boxes decreases. When P1 starts it has max 16 places for second turn. After this it has max of 12 option to play. No of available branch `25 x 24 x 12^11 x 5e8 ~ 3e23`
1. Average branching factor. Find number of branches at each level and divide it by level. Play the game for multiple time. Avg branching is ~8.
1. Time taken to find sol with avg branching factor is `8^25 ~= e22 ~ 1.22 million yrs`.
1. We can limit the depth of our branching to 2 sec (UX). 
1. ![isolation-game](/assets/aind-1/images/depth.jpeg?raw=true)
1. We assume we have 9 level to compute in 2 sec.

## Evaluation Fun & Quiescent Search
1. We need to know at level 9 which player has most boxes to play. We call this evaluation fun and pass it to minimax algo.
1. The result of evaluation fun will change and it depends on the level at which evaluation fun starts. 
1. The idea of quiescence is that the static evaluator should not be applied to positions whose values are unstable, such as those occurring in the middle of the piece trade. In those positions, a small secondary search is conducted until the static evaluation becomes more stable. In games such as chess or checkers, this can be achieved by always exploring any capture moves one level deeper. This extra search is called quiescence search. Applying quiescence search to capture moves quickly will resolve the uncertainties in the position.

