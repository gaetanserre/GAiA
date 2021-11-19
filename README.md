# Deep ViCTORIA

Deep ViCTORIA is a UCI chess engine built with C++ 17.

It performs an in-depth analysis and uses a neural network to evaluate each chess board.

It can perform an analysis in reasonable time between depth 15 and 20+ depending on the number of possible moves.

Its elo rating is 2287 in 10 minutes games ([Lichess](https://lichess.org/) May 2021).

Deep ViCTORIA is not a complete chess program and requires a UCI-compatible graphical user
interface (GUI) (e.g. XBoard with PolyGlot, Scid, Cute Chess, eboard, Arena, Sigma Chess,
Shredder, Chess Partner or Fritz) in order to be used comfortably.

## Build
`cd Engine`

In order to build Deep ViCTORIA, you need [CMake](https://cmake.org/).
Then:
```bash
cd build
cmake ..
make
```

## Usage
```bash
./Deep_VICTORIA
```

## Most used UCI commands:
+ `position startpos [moves move_list]`
+ `position fen your_fen [moves move_list]`
+ `go depth n`
+ `go infinite`: search until you enter `stop`
+ `go movetime t`: search for t milliseconds
+ `go wtime t1 btime t2 [winc t3 binc t4]`: Whites has `t1` ms on clock Blacks has `t2` ms on clock. Whites increment their time by `t3` ms and Blacks increment their time by `t4` ms
+ `go nodes n` search for n nodes (In fact, the number of nodes explored will be a bit greater than *n*)

## Evaluation function
Deep ViCTORIA uses a neural network as an evaluation function.
The goal of this network is to perform a regression to recreate the evaluation function of another chess engine.
You can emulate any of these (I have chosen Stockfish 13, but it could have been Leela or Komodo...)

I used Tensorflow and Python to create and train the neural network.

I used the [Lichess database](https://database.lichess.org) to recover tons of chess games.

### Evaluate positions
To do the regression, I must first evaluate each position of the dataset with a chess engine (here Stockfish 13):

For each position in the dataset, I perform an evaluation with the engine, and I save the result
along the position in a new dataset (the best is to have a command which give the static evaluation of the position
like the `eval` command in Stockfish. If there is no such command, search at depth 1).

### Encode positions
I must encode positions from [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) to numbers:

Each position is encoded as vector of dimension 131:
64 (squares) * 2 (color of the piece + piece type: rook, pawn...) + castlings rights (∈ [0..15]) + en passant square + Whites to play.

I now have the required data to perform a regression

(The evaluation and encoding are done in parallel)

### Structure
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 131)]             0         
_________________________________________________________________
dense (Dense)                (None, 131)               17292     
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8448      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 25,805
Trainable params: 25,805
Non-trainable params: 0
_________________________________________________________________
```
Each *Dense* layer uses `relu` as the activation function except the last one which uses `linear`.

Currently, the network is trained on 58 million positions and tested on 2 million.


Using the [R² metric](https://en.wikipedia.org/wiki/Coefficient_of_determination), the model has a score of about 0.86:

![](ANN/model.jpg)

All the Python and Jupyter files to create the dataset and train the network are available in the `ANN` directory.

## Notes
+ Since Deep ViCTORIA uses a neural network, the evaluation function is much more slower than a 'standard' one. So I needed a very optimized search algorithm and board representation. I first tried to use those from my previous engine [ViCTORIA](https://github.com/Pl4giat01/ViCTORIA) but it was too slow. So I decided to use those from [Stockfish](https://github.com/official-stockfish/Stockfish).
  
+ I used my own implementation of neural network in C++ called [Neural Net](https://github.com/Pl4giat01/NeuralNet).

## Credits
+ [Lichess](https://database.lichess.org/)
+ [Stockfish](https://github.com/official-stockfish/Stockfish)

## License
[GPL v3](https://choosealicense.com/licenses/gpl-3.0/)
