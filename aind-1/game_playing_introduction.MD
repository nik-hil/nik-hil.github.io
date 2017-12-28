# WEEK-2 Game playing with AI

## Game playing with AI is hard. We look for strategies for turn based games

1. Game we pick is Isolation. Player1 starts with 'O'
1. ![isolation game] (https://github.com/nik-hil/nik-hil.github.io/blob/master/aind-1/images/isolation.jpeg?raw=true)
1. Build a tree of all possiblities. Find the branches where player1 will loose. 
1. Teach computer to find loosing branches with a version of [MiniMax algorithm](https://classroom.udacity.com/nanodegrees/nd889/parts/6be67fd1-9725-4d14-b36e-ae2b5b20804c/modules/f719d723-7ee0-472c-80c1-663f02de94f3/lessons/9b1a742a-fa2d-4940-922c-ed426b44f81b/concepts/49461227440923).

## Max number of nodes visited.

1. for 5 x 5 board of isolation game, at each turn number of available branches are , `25 * 24 * 23 * ... * 3 * 2 * 1 = 10^25`. Not feasible.

## Branching factor
1. Consider branching factor. After each player turn no of empty boxes decreases. When P1 starts it has max 16 places for second turn. After this it has max of 12 option to play. No of available branch `25 x 24 x 12^11 x 5e8 ~ 3e23`
1. Average branching factor. Find number of branches at each level and divide it by level. Play the game for multiple time. Avg branching is ~8.
1. Time taken to find sol with avg branching factor is `8^25 ~= e22 ~ 1.22 million yrs`.
1. 
