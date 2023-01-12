# Probabilistic_Drone_Detection
A grid with barriers in which the drone's location must be determined without any prior knowledge of its whereabouts.

## Environment
The drone is capable of moving from cell to cell within the reactor, working on and repairing internal mechanisms. However, while you can issue commands to the sub drone (Up, Down, Left, Right), due to the manner in which you gained access, you don’t have the ability to access the drone’s sensors - you can command it, but you can’t see where it is or get any feedback from it about its position. You are blind. But not helpless.

## Determining the exact location of the drone after a set of commands are executed.
By updating the probability of the drone being in each cell after each move, we can eventually determine its exact location. This process involves continually modifying the probability distribution until there is a 100% chance of the drone being in a specific cell.
