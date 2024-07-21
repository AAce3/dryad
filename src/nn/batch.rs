use arrayvec::ArrayVec;
use ndarray::{ArcArray, ArcArray2, Ix4};
use smallvec::SmallVec;

use crate::{
    mcts::{
        cache::NNEval,
        node::{Edge, WDL},
        params::SearchParams,
        ptrs::NodePtr,
        search_driver::BackpropData,
    },
    movegen::{
        action::{Action, MoveType},
        bitboard,
        board::{Board, Castling},
        history::PositionHistory,
        types::{square, Color, Piece},
    },
};

use super::policy_map::POLICY_MAP;

#[derive(Clone, Copy, Default)]
pub struct BatchEntry {
    pub ptr: Option<NodePtr>,
    pub board: Board,
}

pub struct Batch {
    pub(crate) input_tensor: ArcArray<f32, Ix4>,
    pub(crate) output_policy: ArcArray2<f32>,
    pub(crate) output_wdl: ArcArray2<f32>,
    pub(crate) output_mlh: ArcArray2<f32>,
    entries: Vec<BatchEntry>,
    pub batch_size: usize,
}

impl Batch {
    const NUM_PLANES: usize = 112;

    pub const POLICY_SIZE: usize = 1858;
    pub const WDL_SIZE: usize = 3;
    pub const MLH_SIZE: usize = 1;

    pub fn new(batch_size: usize) -> Self {
        let input_tensor = ArcArray::<f32, Ix4>::zeros((batch_size, Self::NUM_PLANES, 8, 8));
        let output_policy = ArcArray2::<f32>::zeros((batch_size, Self::POLICY_SIZE));
        let output_wdl = ArcArray2::<f32>::zeros((batch_size, Self::WDL_SIZE));
        let output_mlh = ArcArray2::<f32>::zeros((batch_size, Self::MLH_SIZE));
        let entries = Vec::with_capacity(batch_size);

        Self {
            input_tensor,
            output_policy,
            output_wdl,
            output_mlh,
            entries,
            batch_size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn push_board(&mut self, boards: &PositionHistory, ptr: NodePtr) -> Option<()> {
        let batch_idx = self.entries.len();
        if batch_idx == self.batch_size {
            return None;
        }

        self.entries.push(BatchEntry {
            ptr: Some(ptr),
            board: *boards.latest_pos(),
        });

        let batch_size = self.batch_size;
        let buffer = &mut self.input_tensor;
        assert_eq!(buffer.shape(), &[batch_size, 112, 8, 8]);

        let current_board = boards.latest_pos();
        let us = current_board.active_color();
        let them = !us;

        let flipped = us == Color::Black;
        let mut curr_input = buffer.index_axis_mut(ndarray::Axis(0), batch_idx);

        curr_input.fill(0.0);

        let mut planes = curr_input.axis_iter_mut(ndarray::Axis(0));

        // planes 1-104: board history
        for board_idx in 0..8 {
            let board = boards[board_idx];

            for piece_val in 0..6 {
                let mut piece_bb = board.piece_bb(Piece::from(piece_val), us);
                if flipped {
                    piece_bb = piece_bb.swap_bytes();
                }

                let mut plane = planes.next()?;
                while piece_bb != 0 {
                    let square = bitboard::pop_lsb(&mut piece_bb);
                    let file = square::file_of(square) as usize;
                    let rank = square::rank_of(square) as usize;
                    plane[[rank, file]] = 1.0;
                }
            }

            for piece_val in 0..6 {
                let mut piece_bb = board.piece_bb(Piece::from(piece_val), them);
                if flipped {
                    piece_bb = piece_bb.swap_bytes();
                }

                let mut plane = planes.next()?;
                while piece_bb != 0 {
                    let square = bitboard::pop_lsb(&mut piece_bb);
                    let file = square::file_of(square) as usize;
                    let rank = square::rank_of(square) as usize;
                    plane[[rank, file]] = 1.0;
                }
            }

            let num_repetitions = boards.num_repetitions_at(board_idx);
            let mut rep_plane = planes.next()?;
            if num_repetitions != 0 {
                rep_plane.fill(0.0);
            }
        }

        // castling code
        let can_wk = *current_board.castling(Castling::WK);
        let can_bk = *current_board.castling(Castling::BK);
        let can_wq = *current_board.castling(Castling::WQ);
        let can_bq = *current_board.castling(Castling::BQ);

        let (us_kingside, us_queenside, them_kingside, them_queenside) = if us == Color::White {
            (can_wk, can_wq, can_bk, can_bq)
        } else {
            (can_bk, can_bq, can_wk, can_wq)
        };

        // plane 104: can we castle queenside
        planes.next()?.fill((us_queenside as u32) as f32);

        // plane 105: can we castle kingside
        planes.next()?.fill((us_kingside as u32) as f32);

        // plane 106: can they castle queenside
        planes.next()?.fill((them_queenside as u32) as f32);

        // plane 107: can they castle kingside
        planes.next()?.fill((them_kingside as u32) as f32);

        // plane 108: whether it is black to move
        planes.next()?.fill((flipped as u32) as f32);

        // plane 109: percentage of the way to 50 move draw
        let fifty_percentage = current_board.halfmove_clock() as f32 / 100.0;
        planes.next()?.fill(fifty_percentage);

        // plane 110: all zeros
        planes.next()?;

        // plane 111: all ones
        planes.next()?.fill(1.0);

        // plane 112: all zeros

        Some(())
    }

    pub fn pop_output(&mut self, params: &SearchParams) -> BackpropData {
        let output_entry = self.entries.pop().unwrap();
        let board = &output_entry.board;
        let batch_idx = self.entries.len();

        let (wdl_binding, policy_binding, mlh_binding) = (
            self.output_wdl.view(),
            self.output_policy.view(),
            self.output_mlh.view(),
        );

        let (wdl_tensor, policy_tensor, mlh_tensor) = (
            wdl_binding.index_axis(ndarray::Axis(0), batch_idx),
            policy_binding.index_axis(ndarray::Axis(0), batch_idx),
            mlh_binding.index_axis(ndarray::Axis(0), batch_idx),
        );

        // WDL and MLH
        assert!(wdl_tensor.ndim() == 1 && policy_tensor.ndim() == 1 && mlh_tensor.ndim() == 1);
        assert!(
            wdl_tensor.len() == Self::WDL_SIZE
                && policy_tensor.len() == Self::POLICY_SIZE
                && mlh_tensor.len() == Self::MLH_SIZE
        );

        let (win, draw, loss) = (wdl_tensor[[0]], wdl_tensor[[1]], wdl_tensor[[2]]);
        let moves_left = mlh_tensor[[0]];

        // policy softmax over moves
        let legal_moves = board.genmoves();
        let mut max_policy = f32::MIN;
        let mut policy_buffer: ArrayVec<f32, 255> =
            ArrayVec::from_iter(legal_moves.iter().map(|&action| {
                let policy_value = policy_tensor[[action.get_network_idx(board.active_color())]];
                max_policy = max_policy.max(policy_value);
                policy_value
            }));

        let mut total = 0.0;

        // calculate softmax for each of the values in the policy buffer
        for policy_val in policy_buffer.iter_mut() {
            let intermediate_policy =
                ((*policy_val - max_policy) / params.policy_temperature).exp();
            *policy_val = intermediate_policy;
            total += intermediate_policy;
        }

        let scale = if total > 0.0 { 1.0 / total } else { 1.0 };

        // then, scale to 0, 1 range
        let mut edges = SmallVec::from_iter(
            policy_buffer
                .iter()
                .zip(legal_moves.iter())
                .map(|(&policy_val, &action)| Edge::new(policy_val * scale, action)),
        );

        edges.sort_unstable_by(|a: &Edge, b: &Edge| {
            b.get_policy().partial_cmp(&a.get_policy()).unwrap()
        });
        BackpropData {
            eval: NNEval {
                wdl: WDL {
                    win_loss: win - loss,
                    draw,
                    moves_left,
                },
                edges,
            },
            entry: output_entry,
        }
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() == self.batch_size
    }
}

impl Action {
    fn get_network_idx(&self, color: Color) -> usize {
        let from = square::perspective_sqr(self.from(), color) as usize;
        let to = square::perspective_sqr(self.to(), color) as usize;

        let policy_map_idx = match self.move_type() {
            MoveType::Normal | MoveType::Passant => from * 64 + to,
            MoveType::Castle => {
                let rook_square = match to as u8 {
                    square::G1 => square::H1,
                    square::C1 => square::A1,
                    _ => panic!("bad castling"),
                } as usize;
                from * 64 + rook_square
            }
            MoveType::Promotion => {
                if self.pr_piece() == Piece::N {
                    from * 64 + to
                } else {
                    let offset = Piece::Q as usize - (self.pr_piece() as usize); // lc0 index for computing promotions
                    let from_file = square::file_of(from as u8) as usize;
                    let to_file = square::file_of(to as u8) as usize;

                    4096 + ((from_file * 8 + to_file) * 3 + offset)
                }
            }
        };

        POLICY_MAP[policy_map_idx] as usize
    }
}
