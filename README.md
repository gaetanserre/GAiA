# GAiA
GAiA is a UCI chess engine built with C++ 17, [ONNX](https://github.com/microsoft/onnxruntime) and [Tensorflow](https://github.com/tensorflow/tensorflow).

It performs an in-depth analysis and uses a complex squeeze-and-excitation residual network to evaluate each chess board.

It can perform an analysis in reasonable time between depth 10 and 14+ depending on the number of possible moves.

GAiA is not a complete chess program and requires a UCI-compatible graphical user interface (GUI) (e.g. XBoard with PolyGlot, Scid, Cute Chess, eboard, Arena, Sigma Chess, Shredder, Chess Partner or Fritz) in order to be used comfortably.

## Detailed Description
I wrote an article about GAiA which describes in detail its creation process.
You can read it here: [Performing Regression on Complex Data using a
  Squeeze-and-Excitation Residual Neural Network, Chess as a Model System](https://raw.githubusercontent.com/Plagiat01/GAiA/master/article/Performing%20Regression%20on%20Complex%20Data.pdf)

## Build from source
In order to build GAiA, you need [CMake](https://cmake.org/).

GAiA depends on [ONNX](https://github.com/microsoft/onnxruntime) which is an awesome library
for inferring and even training artificial intelligence model. ONNX support many framework
such as CUDA or TensorRT. You need to put the ONNX libraries file in `Engine/build/lib`.
You can find these files [here](https://github.com/microsoft/onnxruntime/releases).

Then,

```bash
cd Engine/build
cmake ..
make
```

## Usage
```bash
./GAiA
```

By default, GAiA is built using the CPU as the execution provider of ONNX because
it was the most efficient on my machine. But you can easily change the EP to CUDA or TensorRT
by changing the variable `EP` in the `CMakeLists`. The accepted values are `CPU`, `CUDA` and `TENSORRT`

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

## Credits
+ [Lichess](https://database.lichess.org/)
+ [Stockfish](https://github.com/official-stockfish/Stockfish)
+ [ONNX](https://github.com/microsoft/onnxruntime)
+ [Tensorflow](https://github.com/tensorflow/tensorflow)

## License
[GPL v3](https://choosealicense.com/licenses/gpl-3.0/)
