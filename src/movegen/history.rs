use crate::mcts::node::Terminal;

use super::{action::Action, board::Board, zobrist::Zobrist};
use anyhow::{anyhow, Result};
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct PositionHistory {
    boards: Vec<Board>,
}

impl Board {
    pub fn parse_uci_move(self, uci_move: &str) -> Result<Action> {
        let list = self.genmoves();
        let action = list
            .iter()
            .find(|&action| action.to_string() == uci_move)
            .ok_or(anyhow!("{uci_move} is not a legal move!"))?;
        Ok(*action)
    }
}

impl PositionHistory {
    pub fn from_uci(fen: &str, moves: &str) -> Result<PositionHistory> {
        let mut start = Board::default();
        start.parse_fen(fen)?;
        let mut history = PositionHistory::fill(&start, 8);
        let mut board = start;
        for uci_move in moves.split_whitespace() {
            let action = board.parse_uci_move(uci_move)?;
            board = board.make_move(action);
            history.boards.push(board);
        }

        Ok(history)
    }

    pub fn from_fen(fen: &str) -> Result<Self> {
        let mut start = Board::default();
        start.parse_fen(fen)?;
        Ok(PositionHistory::fill(&start, 8))
    }

    pub fn fill(board: &Board, count: usize) -> Self {
        let mut history = Vec::with_capacity(count + 32);
        let initial_color = board.active_color();

        let mut new_board = *board;
        for _ in 0..count {
            history.push(new_board);
            new_board.swap_sides();
        }
        history.reverse();
        assert_eq!(new_board.active_color(), initial_color);
        Self { boards: history }
    }

    pub fn num_repetitions_at(&self, index: usize) -> u8 {
        if self.boards.len() < 4 || self[index].halfmove_clock() < 4 {
            return 0;
        }

        let key = self[index].get_hash();
        let halfmove_clock = self[index].halfmove_clock();
        let mut num_repetitions = 0;
        for i in (index..self.boards.len())
            .take(halfmove_clock as usize + 1)
            .step_by(2)
            .skip(1)
        {
            if self[i].get_hash() == key {
                num_repetitions += 1;
            }
        }

        num_repetitions
    }

    pub fn num_repetitions(&self) -> u8 {
        self.num_repetitions_at(0)
    }

    pub fn latest_pos(&self) -> &Board {
        self.boards.last().expect("Empty History")
    }

    pub fn latest_pos_mut(&mut self) -> &mut Board {
        self.boards.last_mut().expect("Empty History")
    }

    pub fn push(&mut self, board: Board) {
        self.boards.push(board);
    }

    pub fn pop(&mut self) -> Option<Board>{
        self.boards.pop()
    }

    pub fn pop_n(&mut self, num_elems: usize) {
        let len = self.boards.len();
        self.boards.truncate(len - num_elems);
    }

    pub fn make_move(&mut self, action: Action) {
        let next_board = self.latest_pos().make_move(action);
        self.push(next_board);
    }

    pub fn unmake_move(&mut self) {
        self.boards.pop();
    }

    pub fn get_hash(&self) -> Zobrist {
        self.latest_pos().get_hash()
    }

    pub fn get_halfmove_hash(&self) -> Zobrist {
        self.latest_pos().get_halfmove_hash()
    }

    pub fn terminal_type(&self) -> Terminal {
        // check for material draw
        let board = self.latest_pos();
        if board.is_material_draw() || self.num_repetitions() >= 2 {
            Terminal::Draw
        } else {
            // check legal moves
            let moves = board.genmoves();
            if moves.is_empty() {
                // check if we are in check
                if board.in_check(board.active_color()) {
                    // mated
                    Terminal::Loss(0)
                } else {
                    Terminal::Draw
                }
            } else {
                // otherwise it's score
                Terminal::Score
            }
        }
    }

    pub fn num_boards(&self) -> usize {
        self.boards.len()
    }

    pub fn clone_from(&mut self, other: &Self) {
        self.boards.clone_from(&other.boards);
    }
}

impl Index<usize> for PositionHistory {
    type Output = Board;

    fn index(&self, index: usize) -> &Self::Output {
        &self.boards[self.boards.len() - 1 - index]
    }
}

impl IndexMut<usize> for PositionHistory {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let last = self.boards.len() - 1;
        &mut self.boards[last - index]
    }
}
