# GAiA
GAiA is a UCI chess engine built with C++ 17 and Tensorflow.

It performs an in-depth analysis and uses a complex squeeze-and-excitation residual network to evaluate each chess board.

It can perform an analysis in reasonable time between depth 10 and 14+ depending on the number of possible moves.

GAiA is not a complete chess program and requires a UCI-compatible graphical user interface (GUI) (e.g. XBoard with PolyGlot, Scid, Cute Chess, eboard, Arena, Sigma Chess, Shredder, Chess Partner or Fritz) in order to be used comfortably.

## Build
In order to build GAiA, you need [CMake](https://cmake.org/).

```bash
cd Engine/build
cmake ..
make
```

## Usage
```bash
./GAiA
```

## Most used UCI commands:
+ `position startpos [moves move_list]`
+ `position fen your_fen [moves move_list]`
+ `go depth n`
+ `go infinite`: search until you enter `stop`
+ `go movetime t`: search for t milliseconds
+ `go wtime t1 btime t2 [winc t3 binc t4]`: Whites has `t1` ms on clock Blacks has `t2` ms on clock. Whites increment their time by `t3` ms and Blacks increment their time by `t4` ms
+ `go nodes n` search for n nodes (In fact, the number of nodes explored will be a bit greater than *n*)

## Notebooks
All the notebooks and python files used to build GAiA's network are available in
the `SE-ResNet` directory.

Notebooks order:
1. `encode.ipynb`
2. `choose_hyperparameters.ipynb`
3. `train_model.ipynb`
4. `results.ipynb`

## Detailed Description
I wrote a paper about GAiA which describes in detail its creation process.
You can read it here: [Performing Regression on Complex Data using a
  Squeeze-and-Excitation Residual Neural Network, Chess as a Model System](paper/Performing%20Regression%20on%20Complex%20Data.pdf)

## Credits
+ [Lichess](https://database.lichess.org/)
+ [Stockfish](https://github.com/official-stockfish/Stockfish)
+ [frugally-deep](https://github.com/Dobiasd/frugally-deep)

## License
[GPL v3](https://choosealicense.com/licenses/gpl-3.0/)