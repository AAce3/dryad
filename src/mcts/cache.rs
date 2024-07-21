use smallvec::SmallVec;

use crate::movegen::zobrist::Zobrist;

use super::node::{Edge, WDL};

pub struct NNCache {
    data: Vec<NNCacheEntry>,
}

#[derive(Clone)]
pub struct NNEval {
    pub wdl: WDL,
    pub edges: SmallVec<[Edge; 64]>,
}


impl NNCache {
    pub fn new(capacity: usize) -> Self {
        let capacity = usize::max(capacity, 1);
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(NNCacheEntry::default());
        }
        Self { data }
    }

    pub fn probe(&self, hash: Zobrist) -> Option<NNEval> {
        let elem = &self.data[hash as usize % self.data.len()];
        if elem.hash == hash {
            Some(NNEval::from(elem))
        } else {
            None
        }
    }

    pub fn insert(&mut self, hash: Zobrist, output: &NNEval) {
        let entry = NNCacheEntry::new(hash, output);
        let len = self.data.len();
        self.data[hash as usize % len] = entry;
    }

    pub fn clear(&mut self) {
        for entry in self.data.iter_mut() {
            *entry = NNCacheEntry::default()
        }
    }

    pub fn resize(&mut self, new_size: usize) {
        self.data.resize_with(new_size, NNCacheEntry::default)
    }
}

#[derive(Default)]
pub struct NNCacheEntry {
    win_loss: f32,
    draw_prob: f32,
    moves_left: f32,
    edges: Box<[Edge]>,
    hash: Zobrist,
}

impl NNCacheEntry {
    pub fn new(zobrist: Zobrist, output: &NNEval) -> Self {
        Self {
            win_loss: output.wdl.win_loss,
            draw_prob: output.wdl.draw,
            moves_left: output.wdl.moves_left,
            edges: Box::from_iter(output.edges.iter().copied()),
            hash: zobrist,
        }
    }
}

impl From<&NNCacheEntry> for NNEval {
    fn from(value: &NNCacheEntry) -> Self {
        Self {
            wdl: WDL {
                win_loss: value.win_loss,
                draw: value.draw_prob,
                moves_left: value.moves_left,
            },
            edges: SmallVec::from_iter(value.edges.as_ref().iter().copied()),
        }
    }
}
