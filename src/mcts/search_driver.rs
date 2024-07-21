use std::{
    sync::Arc,
    thread::{self, JoinHandle},
    time::Instant,
};

use crossbeam::{
    atomic::AtomicCell,
    channel::{self, Receiver, Sender},
};
use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::{
    movegen::{action::Action, history::PositionHistory},
    nn::{
        batch::{Batch, BatchEntry},
        network::Network,
    },
};

use super::{
    cache::{NNCache, NNEval},
    params::SearchParams,
    ptrs::NodePtr,
    search::{process_backprop_request, Searcher, SelectError},
    time_manager::TimeManager,
    tree::NodeTree,
};

// A general overview of multithreaded structure:
// There are two primary threads. These threads alternate between performing NN computations and gathering,
// which allows for overlapping execution. Each of these workers enlist the help of the global threadpool.
// For example, it might initially look something like this, with Thread 1 performing search and Thread 2 running
// a blocking evaluation on the neural network:
//
//
//     +----------+                            +----------+
//     | Thread 1 | ~~~~   [ search ]          | Thread 2 | ~~~~ [ Neural Network ]
//     +----------+          vvvv              +----------+          ^^^^^
//                 \_____( [ Batch ] )                     \______( [ Batch ] )
//                            ^     ^
//                            |      \
//                            |       \
//                     +----------+   |
//                     | Worker 1 |   |
//                     +----------+   |
//                                    +----------+
//                                    | Worker 2 |
//                                    +----------+
//
// Once Thread 2 completes its evaluation and Thread 1 completes its search, Thread 1 gets the NN lock and Thread 2 gets the searcher lock.
//                                 _______________________________________
//                               /                                        \
//                               |                                        |
//     +----------+              v                +----------+            v
//     | Thread 1 | ~~~~ [ Neural Network ]       | Thread 2 | ~~~~    [ search ]
//     +----------+          ^^^^^                +----------+           vvvv
//                 \_____( [ Batch ] )                        \______( [ Batch ] )
//                                                                   ^    ^
//                                                                  /     |
//                                                                 /      |
//                     +----------+-------------------------------        |
//                     | Worker 1 |                                       |
//                     +----------+                                       |
//                                    +----------+------------------------
//                                    | Worker 2 |
//                                    +----------+

pub type NodePath = SmallVec<[Action; 64]>;

pub type Flag = AtomicCell<bool>;

pub struct SharedTreeData {
    pub tree: NodeTree,
    pub search_stats: SearchStatistics,
    pub params: SearchParams,
    pub cache: Mutex<NNCache>,
}

impl SharedTreeData {
    pub fn clear(&self) {
        self.tree.clear();
        self.search_stats.clear();
        self.cache.lock().clear();
    }

    pub fn initialize_root(&self, network: &Arc<dyn Network>, position: &PositionHistory) {
        let params = &self.params;
        let mut default_batch = Batch::new(params.batch_size as usize);

        let (root_ptr, mut root_node) = self.tree.allocate_node().unwrap();

        default_batch.push_board(position, NodePtr::from(root_ptr));
        network.locked_compute(&mut default_batch).unwrap();

        let result = default_batch.pop_output(params);
        root_node.full_visits += 1;

        root_node.set_nn_eval(&result.eval);
       
        self.tree.set_root(root_ptr);
        self.cache
            .lock()
            .insert(result.entry.board.get_hash(), &result.eval);
    }
}

#[derive(Default)]
pub struct SearchStatistics {
    pub batches_sent: AtomicCell<u32>,
    pub selective_depth: AtomicCell<u8>,
    pub cumulative_depth: AtomicCell<u64>,
    pub node_count: AtomicCell<u32>,
}

impl SearchStatistics {
    pub fn avg_depth(&self) -> u8 {
        (self.cumulative_depth.load() / u64::max(self.node_count.load() as u64, 1)) as u8
    }

    pub fn clear(&self) {
        self.batches_sent.store(0);
        self.selective_depth.store(0);
        self.cumulative_depth.store(0);
        self.node_count.store(0);
    }
}

#[derive(Default)]
pub struct BatchStatistics {
    pub num_collisions: AtomicCell<u16>,
    pub batch_count: AtomicCell<u16>,
}

impl BatchStatistics {
    pub fn reset(&self) {
        self.num_collisions.store(0);
        self.batch_count.store(0)
    }
}

pub struct SearchTask {
    pub num_visits: u16,
    pub depth: u8,
    pub ptr: NodePtr,
    pub nodepath: NodePath,
    pub batch: Arc<Mutex<Batch>>,
}

#[derive(Clone)]
pub struct BackpropData {
    pub eval: NNEval,
    pub entry: BatchEntry,
}

impl SearchTask {
    pub fn default_task(tree_data: &SharedTreeData, batch: Arc<Mutex<Batch>>) -> Self {
        Self {
            num_visits: tree_data.params.default_selection_size(),
            depth: 0,
            ptr: tree_data.tree.root(),
            nodepath: SmallVec::new(),
            batch,
        }
    }

    pub fn get_num_visits(
        &mut self,
        tree_data: &SharedTreeData,
        batch_stats: &BatchStatistics,
    ) -> u16 {
        let num_visits = tree_data.params.next_search_size(batch_stats);
        self.num_visits = tree_data.params.next_search_size(batch_stats);

        self.ptr = tree_data.tree.root();
        self.depth = 0;
        self.nodepath.clear();
        num_visits
    }
}

pub struct ThreadPool {
    task_sender: Sender<SearchTask>,
    result_receiver: Receiver<Result<(), SelectError>>,
    num_tasks: Arc<AtomicCell<u16>>,
}

pub struct SearcherThread {
    searcher: Searcher,
    network: Arc<dyn Network>,
    threadpool: Arc<Mutex<ThreadPool>>,
    batch: Arc<Mutex<Batch>>,
    stop_flag: Arc<Flag>,
    time_manager: Arc<Mutex<TimeManager>>,
}

#[derive(Clone)]
pub struct Worker {
    searcher: Searcher,
    task_receiver: Receiver<SearchTask>,
    task_sender: Sender<SearchTask>,
    result_sender: Sender<Result<(), SelectError>>,
    num_tasks: Arc<AtomicCell<u16>>,
    stop_flag: Arc<Flag>,
}

pub fn create_threads(
    tree: &Arc<SharedTreeData>,
    network: &Arc<dyn Network>,
    history: &mut PositionHistory,
    time_manager: Arc<Mutex<TimeManager>>,
    stop_flag: Arc<Flag>,
) -> (SearcherThread, Vec<JoinHandle<()>>) {
    let num_workers = tree.params.num_workers as usize;
    let (selection_sender, selection_receiver) = channel::bounded(num_workers * 32);
    let (result_sender, result_receiver) = channel::bounded(num_workers);

    let threadpool = ThreadPool {
        task_sender: selection_sender.clone(),
        result_receiver,
        num_tasks: Arc::new(Default::default()),
    };

    let batch = Arc::new(Mutex::new(Batch::new(tree.params.batch_size as usize)));

    let searcher = Searcher {
        history: history.clone(),
        shared_tree: Arc::clone(tree),
        batch_stats: Arc::new(BatchStatistics::default()),
    };

    let mut handles = vec![];

    for _ in 0..num_workers {
        let mut search_worker = Worker {
            searcher: searcher.clone(),
            task_receiver: selection_receiver.clone(),
            task_sender: selection_sender.clone(),
            result_sender: result_sender.clone(),
            num_tasks: Arc::clone(&threadpool.num_tasks),
            stop_flag: Arc::clone(&stop_flag),
        };
        let handle = thread::spawn(move || search_worker.wait_for_tasks());
        handles.push(handle);
    }

    let main_searcher = SearcherThread {
        searcher: searcher.clone(),
        network: Arc::clone(network),
        threadpool: Arc::new(Mutex::new(threadpool)),
        stop_flag: stop_flag.clone(),
        batch,
        time_manager,
    };

    (main_searcher, handles)
}

pub const REPORT_INTERVAL: u32 = 8;

impl SearcherThread {
    pub fn search_loop(&mut self) {
        // only one thread is allowed to report.
        let timeman_lock = self.time_manager.try_lock();
        let mut highest_depth = 0;
        let mut highest_seldepth = 0;
        let mut last_report = Instant::now();
        loop {
            // only one thread has a lock on the threadpool at one time
            let threadpool = self.threadpool.lock();

            // try to report if we are holding onto the threadpool. We don't want anyone else trying to traverse the tree
            // while we're reading from it.
            if let Some(ref time_manager) = timeman_lock {
                let search_stats = &self.searcher.shared_tree.search_stats;
                let should_stop = time_manager.should_stop(search_stats);
                if should_stop {
                    self.stop_flag.store(true);
                    break;
                } else {
                    let seldepth = search_stats.selective_depth.load();
                    let avg_depth = search_stats.avg_depth();
                    let mut should_report = false;

                    if seldepth > highest_seldepth || avg_depth > highest_depth {
                        highest_seldepth = seldepth;
                        highest_depth = avg_depth;
                        should_report = true;
                    } else {
                        // The depth in MCTS engines doesn't go up as fast as in AB engines. Therefore, we should report every
                        // n seconds to avoid keeping the user waiting
                        let elapsed = last_report.elapsed();
                        if elapsed.as_secs() > (REPORT_INTERVAL as u64) {
                            should_report = true;
                        }
                    }

                    if should_report {
                        last_report = Instant::now();
                        let time = time_manager.get_time_millis();
                        if time != 0 {
                            println!("{}", self.searcher.shared_tree.report(time));
                        }
                    }
                }
            }

            let mut errored = false;

            let searcher = &mut self.searcher;
            let mut task = SearchTask::default_task(&searcher.shared_tree, Arc::clone(&self.batch));

            loop {
                let mut selection_result =
                    searcher.multi_select(&mut task, &threadpool.task_sender());
                // wait for spawned worker threads to finish processing
                loop {
                    if threadpool.num_tasks.load() == 0 {
                        break;
                    }

                    match threadpool.result_receiver.try_recv() {
                        Ok(msg) => {
                            selection_result = selection_result.or(msg);
                        }
                        Err(channel::TryRecvError::Empty) => (),
                        Err(channel::TryRecvError::Disconnected) => panic!("Channel Disconnected!"),
                    }
                }

                if selection_result.is_err() {
                    errored = true;
                    break;
                }
                // If we completed the search with no issues, start a new search iteration
                else {
                    let next_search_visits =
                        task.get_num_visits(&searcher.shared_tree, &searcher.batch_stats);
                    // if the tree is full, then stop the search
                    if searcher.shared_tree.tree.remaining_nodes() < next_search_visits as u32 {
                        errored = true;
                        break;
                    }

                    if (self.network.is_unlocked() && self.batch.lock().len() > 16)
                        || next_search_visits == 0
                    {
                        // Otherwise, if we've filled our batch, reached the max number of collisions, or the nn is available,
                        // send to nn for eval
                        break;
                    }
                }
            }

            drop(threadpool);

            searcher.batch_stats.reset();
            self.evaluate_and_backprop();
            if self.stop_flag.load() || errored {
                break;
            }
        }

        if timeman_lock.is_some() {
            // only have the time manager thread report bestmove
            let tree = &self.searcher.shared_tree.tree;
            let root_node = tree.get(tree.root());
            let best_child = root_node
                .children(tree)
                .into_iter()
                .max_by_key(|a| a.num_visits())
                .unwrap();
            println!("bestmove {}", best_child.parent_edge.get_action());
        }
    }

    fn evaluate_and_backprop(&self) {
        let searcher = &self.searcher;
        let mut batch = self.batch.lock();
        self.network
            .locked_compute(&mut batch)
            .expect("Failed neural network computation");

        let shared_tree = &searcher.shared_tree;

        while !batch.is_empty() {
            let output = batch.pop_output(&shared_tree.params);
            shared_tree
                .cache
                .lock()
                .insert(output.entry.board.get_hash(), &output.eval);
            process_backprop_request(&output, &shared_tree.tree);
        }
    }
}

impl Worker {
    fn wait_for_tasks(&mut self) {
        loop {
            let task = self.task_receiver.try_recv();
            match task {
                Ok(mut task) => {
                    let num_moves_added = task.nodepath.len();
                    for action in &task.nodepath {
                        self.searcher.history.make_move(*action);
                    }
                    let searcher = &mut self.searcher;
                    let sender = TaskScheduler {
                        task_sender: &self.task_sender,
                        num_tasks: &self.num_tasks,
                    };
                    let search_result = searcher.multi_select(&mut task, &sender);

                    self.searcher.history.pop_n(num_moves_added);

                    self.result_sender
                        .send(search_result)
                        .expect("Channel disconnected!");
                    self.num_tasks.fetch_sub(1);
                }
                Err(channel::TryRecvError::Empty) => (),
                Err(channel::TryRecvError::Disconnected) => panic!("Channel Disconnected!"),
            }

            if self.stop_flag.load() && self.num_tasks.load() == 0 {
                break;
            }
        }
    }
}

// task sender used during search
pub struct TaskScheduler<'a> {
    task_sender: &'a Sender<SearchTask>,
    num_tasks: &'a Arc<AtomicCell<u16>>,
}

impl<'a> TaskScheduler<'a> {
    pub fn send_task(&self, task: SearchTask) {
        self.task_sender.send(task).expect("Channel disconnected!");
        self.num_tasks.fetch_add(1);
    }
}

impl ThreadPool {
    pub fn task_sender(&self) -> TaskScheduler {
        TaskScheduler {
            task_sender: &self.task_sender,
            num_tasks: &self.num_tasks,
        }
    }
}

impl Clone for SearcherThread {
    fn clone(&self) -> Self {
        Self {
            searcher: self.searcher.clone(),
            network: self.network.clone(),
            threadpool: self.threadpool.clone(),
            stop_flag: self.stop_flag.clone(),
            batch: Arc::new(Mutex::new(Batch::new(
                self.searcher.shared_tree.params.batch_size as usize,
            ))),
            time_manager: self.time_manager.clone(),
        }
    }
}
