use super::{board::Board, zobrist::Zobrist};
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct PerftEntry {
    key: Zobrist,
    depth: u8,
    num_nodes: u64,
}

impl Board {
    pub fn divide_perft(&mut self, depth: u8) {
        let start_list = self.genmoves();
        let mut total_nodes = 0;
        let start = Instant::now();
        for action in start_list.iter() {
            let mut board = self.make_move(*action);

            let perft = board.perft(depth - 1);

            total_nodes += perft;
            println!("{} {perft}", *action);
        }
        let end = Instant::now();

        let elapsed = end.duration_since(start).as_secs_f32();
        print!("\nPerft: found {total_nodes} in {elapsed} seconds");
        println!(" ({} nps)", ((total_nodes as f32) / elapsed) as u64);
    }

    pub fn hashed_divide_perft(&mut self, depth: u8, hash_size: usize) {
        let mut hashtable = vec![PerftEntry::default(); hash_size];
        let start_list = self.genmoves();
        let mut total_nodes = 0;

        for action in start_list.iter() {
            let mut board = self.make_move(*action);
            let perft = board.hashed_perft(depth - 1, &mut hashtable);

            total_nodes += perft;
            println!("{} {perft}", *action);
        }
        println!("\n{}", total_nodes);
    }

    fn hashed_perft(&mut self, depth: u8, hashtable: &mut [PerftEntry]) -> u64 {
        if depth == 0 {
            return 1;
        }
        let len = hashtable.len();
        let perft_entry = &mut hashtable[self.get_hash() as usize % len];
        if self.get_hash() == perft_entry.key && depth == perft_entry.depth {
            perft_entry.num_nodes
        } else {
            let mut nodes = 0;
            let list = self.genmoves();
            if depth == 1 {
                return list.len() as u64;
            }

            for action in list.iter() {
                let mut board = self.make_move(*action);
                nodes += board.perft(depth - 1);
            }

            perft_entry.num_nodes = nodes;
            perft_entry.key = self.get_hash();
            perft_entry.depth = depth;
            nodes
        }
    }
    fn perft(&mut self, depth: u8) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut nodes = 0;
        let list = self.genmoves();

        if depth == 1 {
            return list.len() as u64;
        }

        for action in list.iter() {
            let mut board = self.make_move(*action);
            nodes += board.perft(depth - 1);
        }

        nodes
    }
}
