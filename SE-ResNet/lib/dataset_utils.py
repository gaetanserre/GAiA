import numpy as np
import chess
import h5py

def get_castling_rights(board):
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    
    res = 0
    coeff = 1
    for r in rights:
        if r:
            res += coeff
        coeff *= 2
    
    return res

def piece_to_vec(piece, color):
  offset = 0 if color else 6
  res = np.zeros(12)
  res[offset+piece - 1] = 1
  return res

nb_channels = 15
board_shape = (8, 8, nb_channels)

def encode_position(fen):
  res = np.zeros((8, 8, nb_channels), dtype=np.float32)
  board = chess.Board(fen)
  for rank in range(8):
    for file in range(8):
      square = (rank * 8) + file
      piece_type = board.piece_type_at(square)
      if piece_type is None:
        continue
      res[rank, file, :12] = piece_to_vec(piece_type, board.color_at(square))
  res[:, :, 12] = board.turn
  res[:, :, 13] = get_castling_rights(board)
  res[:, :, 14] = -1 if board.ep_square is None else board.ep_square
  return res



def store_many_hdf5(images, labels, directory, tag=""):
  num_images = len(images)

  # Create a new HDF5 file
  file = h5py.File(directory + f"{num_images}_position{tag}.h5", "w")

  # Create a dataset in the file
  dataset = file.create_dataset(
      "images", np.shape(images), data=images
  )
  meta_set = file.create_dataset(
      "label", np.shape(labels), data=labels
  )
  file.close()

def read_many_hdf5(num_images, directory, tag=""):
  file = h5py.File(directory + f"{num_images}_position{tag}.h5", "r+")

  images = np.array(file["/images"])
  labels = np.array(file["/label"])

  file.close()

  return images, labels