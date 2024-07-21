use arrayvec::ArrayVec;
use core::fmt;
use parking_lot::Mutex;
use std::{
    fmt::Display,
    fs,
    io::{self, BufRead},
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::{
    mcts::{
        cache::NNCache,
        node::Terminal,
        params::{SearchParams, TreeParams},
        search_driver::{create_threads, Flag, SearchStatistics, SharedTreeData},
        time_manager::TimeManager,
        tree::NodeTree,
    },
    movegen::{action::Action, board::Board, history::PositionHistory, types::Color},
    nn::{network::Network, onnx_tensorrt::OnnxTRTNetwork},
    VERSION,
};
use std::fmt::Write;

pub fn uci_loop(network_path: Option<&str>) {
    let mut tree_params = TreeParams::default();
    let search_params = SearchParams::default();

    let mut position =
        PositionHistory::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();

    let mut tree_data = Arc::new(SharedTreeData {
        tree: NodeTree::new(tree_params.max_tree_size),
        search_stats: SearchStatistics::default(),
        params: search_params,
        cache: Mutex::new(NNCache::new(tree_params.cache_size)),
    });

    println!("Initializing network! This may take awhile. Do not exit the program!");
    let network_initialization_start = Instant::now();

    let network: Arc<dyn Network> = match find_network(network_path) {
        Some(network) => network,
        None => {
            eprintln!("There was a problem loading the network. Please ensure that the path you have entered is a valid path and that all dependencies are available on your system. Exiting process.");

            thread::sleep(Duration::from_secs(1));
            return;
        }
    };

    tree_data.initialize_root(&network, &position);
    let network_initialization_end = Instant::now();
    let time = network_initialization_end.duration_since(network_initialization_start);
    println!(
        "Network initialized successfully in {} seconds! Ready to receive uci input.",
        time.as_secs()
    );

    let stop_flag: Arc<Flag> = Arc::new(Flag::default());

    let mut io_string = String::new();

    loop {
        io_string.clear();

        io::stdin()
            .lock()
            .read_line(&mut io_string)
            .expect("Failed to read stdin");

        if io_string.starts_with("uci") {
            println!("id name dryad {VERSION}");
            println!("id author Aaron Li");
            print_options();
            println!("uciok");
        } else if io_string.starts_with("isready") {
            println!("readyok");
        } else if io_string.starts_with("setoption") {
            if let Some((name, value)) = parse_setoption(&io_string) {
                match name.as_str() {
                    "MaxTreeSize" => {
                        let new_size = value.parse().ok();
                        if let Some(new_size) = new_size {
                            if new_size != tree_params.max_tree_size {
                                let tree = Arc::get_mut(&mut tree_data).unwrap();
                                tree.tree = NodeTree::new(new_size);
                                tree.cache.lock().clear();
                                tree.search_stats.clear();
                                tree_params.max_tree_size = new_size;
                            }
                        }
                    }

                    "CacheSize" => {
                        let new_size = value.parse().ok();
                        if let Some(new_size) = new_size {
                            if new_size != tree_params.cache_size {
                                let mut cache = tree_data.cache.lock();
                                cache.clear();
                                cache.resize(new_size)
                            }
                        }
                    }

                    "NumWorkers" => {
                        let new_size = value.parse().ok();
                        if let Some(new_size) = new_size {
                            let tree = Arc::get_mut(&mut tree_data).unwrap();
                            tree.params.num_workers = new_size
                        }
                    }

                    "BatchSize" => {
                        let new_size = value.parse().ok();
                        if let Some(new_size) = new_size {
                            let tree = Arc::get_mut(&mut tree_data).unwrap();
                            tree.params.batch_size = new_size;
                        }
                    }
                    "MoveOverhead" => {
                        let new_overhead = value.parse().ok();
                        if let Some(new_size) = new_overhead {
                            tree_params.move_overhead_ms = new_size;
                        }
                    }
                    "Clear Tree" => tree_data.clear(),
                    _ => (),
                }
            }
        } else if io_string.starts_with("stop") {
            stop_flag.store(true);
        } else if io_string.starts_with("position") {
            let new_position = parse_position(&io_string);
            assert!(new_position.is_some());
            if let Some(mut new_position) = new_position {
                let new_root = tree_data
                    .tree
                    .try_prune(&mut position, &mut new_position, 2);
                position = new_position;

                if new_root.is_none() {
                    // we cannot reuse the tree. Therefore, reinitialize it.
                    tree_data.initialize_root(&network, &position);
                }
            }
        } else if io_string.starts_with("go") {
            tree_data.search_stats.clear();
            let time_manager = parse_go_command(&io_string, position.latest_pos());
            if let Some(time_manager) = time_manager {
                stop_flag.store(false);

                let (mut search_thread_1, _handles) = create_threads(
                    &tree_data,
                    &network,
                    &mut position,
                    Arc::new(Mutex::new(time_manager)),
                    stop_flag.clone(),
                );

                let mut search_thread_2 = search_thread_1.clone();
                let _handle_1 = thread::spawn(move || search_thread_1.search_loop());
                let _handle_2 = thread::spawn(move || search_thread_2.search_loop());
            }
        } else if io_string.starts_with("quit") {
            return;
        }
    }
}

struct UciOption {
    name: &'static str,
    option: UciOptionType,
}

impl Display for UciOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "option name {} ", self.name)?;
        match &self.option {
            UciOptionType::Spin { default, min, max } => {
                write!(f, "type spin default {default} min {min} max {max}")?
            }
            UciOptionType::Check { default } => write!(f, "type check default {default}")?,
            UciOptionType::Combo { default, options } => {
                write!(f, "type combo default {default}")?;
                for option in options {
                    write!(f, " var {option}")?;
                }
            }
            UciOptionType::String { default } => write!(f, "type string default {default}")?,
            UciOptionType::Button => write!(f, "type button")?,
        }
        Ok(())
    }
}

#[allow(dead_code)]
enum UciOptionType {
    Spin {
        default: i64,
        min: i64,
        max: i64,
    },
    Check {
        default: bool,
    },
    Combo {
        default: &'static str,
        options: Vec<&'static str>,
    },
    String {
        default: &'static str,
    },
    Button,
}

fn print_options() {
    let tree_params = TreeParams::default();
    let search_params = SearchParams::default();

    let available_options = [
        UciOption {
            name: "MaxTreeSize",
            option: UciOptionType::Spin {
                default: tree_params.max_tree_size as i64,
                min: 1_000,
                max: 4_000_000_000,
            },
        },
        UciOption {
            name: "CacheSize",
            option: UciOptionType::Spin {
                default: tree_params.cache_size as i64,
                min: 0,
                max: 4_000_000_000,
            },
        },
        UciOption {
            name: "NumWorkers",
            option: UciOptionType::Spin {
                default: search_params.num_workers as i64,
                min: 0,
                max: 256,
            },
        },
        UciOption {
            name: "BatchSize",
            option: UciOptionType::Spin {
                default: search_params.batch_size as i64,
                min: 1,
                max: 4096,
            },
        },
        UciOption {
            name: "MoveOverhead",
            option: UciOptionType::Spin {
                default: tree_params.move_overhead_ms as i64,
                min: 0,
                max: 5000,
            },
        },
        UciOption {
            name: "Clear Tree",
            option: UciOptionType::Button,
        },
    ];

    for option in available_options {
        println!("{}", option);
    }
}

fn parse_setoption(command: &str) -> Option<(String, String)> {
    let tokens = command.split_whitespace();

    let mut name = String::new();
    let mut value = String::new();

    let mut curr_string = None;

    for token in tokens.skip(1) {
        match token {
            "name" => curr_string = Some(&mut name),
            "value" => curr_string = Some(&mut value),
            _ => curr_string.as_deref_mut()?.push_str(token),
        }
    }

    Some((name, value))
}

fn parse_position(command: &str) -> Option<PositionHistory> {
    let mut fen = String::new();
    let mut moves = String::new();

    let mut curr_string = None;

    for token in command.split_whitespace().skip(1) {
        match token {
            "startpos" => {
                fen.push_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                curr_string = None;
            }
            "fen" => {
                curr_string = Some(&mut fen);
            }
            "moves" => curr_string = Some(&mut moves),
            _ => {
                curr_string.as_deref_mut()?.push_str(token);
                curr_string.as_deref_mut()?.push(' ');
            }
        }
    }

    PositionHistory::from_uci(&fen, &moves).ok()
}

fn parse_go_command(command: &str, board: &Board) -> Option<TimeManager> {
    let mut wtime = None;
    let mut winc = None;
    let mut btime = None;
    let mut binc = None;
    let mut max_depth = None;
    let mut max_nodes = None;
    let mut move_time = None;
    let mut moves_to_go = None;
    let mut perft_depth = None;

    let mut parse_target = None;

    for token in command.split_whitespace().skip(1) {
        match token {
            "wtime" => parse_target = Some(&mut wtime),
            "winc" => parse_target = Some(&mut winc),
            "btime" => parse_target = Some(&mut btime),
            "binc" => parse_target = Some(&mut binc),
            "depth" => parse_target = Some(&mut max_depth),
            "nodes" => parse_target = Some(&mut max_nodes),
            "movetime" => parse_target = Some(&mut move_time),
            "infinite" => (),
            "ponder" => (),
            "mate" => (),
            "movestogo" => parse_target = Some(&mut moves_to_go),
            "perft" => parse_target = Some(&mut perft_depth),

            _ => {
                *parse_target? = Some(token.parse::<u64>().ok()?);
                parse_target = None;
            }
        }
    }

    if let Some(perft_depth) = perft_depth {
        board.clone().divide_perft(perft_depth as u8);
        return None;
    }

    let (time, increment) = match board.active_color() {
        Color::White => (wtime, winc),
        Color::Black => (btime, binc),
    };

    Some(TimeManager::new(
        time,
        increment,
        move_time,
        max_nodes.map(|a| a as u32),
        max_depth.map(|a| a as u8),
    ))
}

pub fn find_network(path: Option<&str>) -> Option<Arc<dyn Network>> {
    const NETWORK_FP_EXTENSION: &str = "onnx";
    if let Some(path) = path {
        return Some(Arc::new(OnnxTRTNetwork::new(path, 0).ok()?));
    } else {
        let entries = fs::read_dir(".").ok()?;
        for entry in entries.flatten() {
            if let Some(file_name) = entry.file_name().to_str() {
                if Path::new(file_name)
                    .extension()
                    .map_or(false, |ext| ext == NETWORK_FP_EXTENSION)
                {
                    if let Ok(network) = OnnxTRTNetwork::new(file_name, 0) {
                        return Some(Arc::new(network));
                    }
                }
            }
        }
    }
    None
}

enum UciScore {
    Centipawns(i32),
    Mate(i32),
}

impl fmt::Display for UciScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UciScore::Centipawns(cp) => write!(f, "cp {}", cp),
            UciScore::Mate(mate) => write!(f, "mate {}", mate),
        }
    }
}

impl SharedTreeData {
    pub fn report(&self, time_ms: u64) -> String {
        let mut string = String::new();
        let mut pv: ArrayVec<Action, 255> = ArrayVec::new();
        let mut nodeptr = self.tree.root();

        let statistics = &self.search_stats;
        let node_count = self.search_stats.node_count.load();

        let depth = statistics.cumulative_depth.load() / (u32::max(node_count, 1) as u64);
        let seldepth = statistics.selective_depth.load();

        let root_node = self.tree.get(nodeptr);

        let score = match root_node.terminal_state {
            Terminal::Score => UciScore::Centipawns(q_to_cp(root_node.calculate_q())),
            Terminal::Win(plies) => UciScore::Mate(plies as i32 / 2),
            Terminal::Loss(plies) => UciScore::Mate(-(plies as i32) / 2),
            Terminal::Draw => UciScore::Centipawns(0),
        };

        drop(root_node);
        loop {
            let node = self.tree.get(nodeptr);

            let children = node.children(&self.tree);
            let best_child = children.iter().max_by_key(|a| a.num_visits());
            if let Some(child) = best_child {
                if let Some(child_ptr) = child.ptr {
                    pv.push(child.parent_edge.get_action());
                    nodeptr = child_ptr;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let nps = node_count as u64 * 1000 / time_ms;

        write!(
            string,
            "info depth {depth} seldepth {seldepth} time {time_ms} score {score} nodes {node_count} nps {nps} pv"
        )
        .unwrap();
        for action in pv {
            write!(string, " {}", action).unwrap();
        }
        string
    }
}
// centipawn conversion formula from lc0
fn q_to_cp(q: f32) -> i32 {
    (90.0 * f32::tan(1.563_754_2 * q)) as i32
}
