use std::sync::Arc;

use arrayvec::ArrayVec;
use parking_lot::Mutex;

use crate::{
    mcts::node::Terminal,
    movegen::{action::Action, history::PositionHistory},
    nn::batch::Batch,
};

use super::{
    node::{weighted_avg, ChildrenList, Edge, TempNode, WDL},
    params::SearchParams,
    ptrs::{NodeGuard, NodePtr},
    search_driver::{BackpropData, BatchStatistics, SearchTask, SharedTreeData, TaskScheduler},
    tree::NodeTree,
};

// The goal of every search is to fill a batch and send it to the neural network for computation. However, because
// this batched flavor of MCTS doesn't get immediate updates to backpropogate back up to the tree, we can save a lot of
// cpu time by trying to fill the entire batch in one go, selecting multiple nodes at the same time.
// In every node, we precalculate how many of the visits will go to each child, then recursively visit those children.
//
//                              Parent: 32 visits (batch size)
//                                         |
//                       +-----------------+----------------+
//                       |                 |                |
//                Child 1: 5 visits  Child 2: 15 visits   Child 3: 12 visits
//                       |                 |                 |
//                      ...               ...               ...
//
// This also makes multithreading a lot easier, as we can distribute child visits across helper threads.
//
//
// However, there are a few caveats: We don't do "virtual loss," which treats unevaluated nodes as "losing," because that leads
// to poor node quality. Instead, we do "virtual visits," which rely on the UCB formula, which favors under-visited children.
// This allows us to not always select the same node. However, this is prone to selecting the same node many times before it
// switches away.
// When we have multiple visits to an unevaluated node, we call those "collisions." When this happens, we treat
// one neural network evaluation as if it were multiple, appropriately weighting it during backpropogation.
// We obviously don't want this to happen very often, so we limit the number of collisions.

#[derive(Debug)]
pub enum SelectError {
    TreeFull,
    TerminalRoot,
}

#[derive(Clone)]
pub struct Searcher {
    pub history: PositionHistory,
    pub shared_tree: Arc<SharedTreeData>,
    pub batch_stats: Arc<BatchStatistics>,
}

impl Searcher {
    pub fn multi_select(
        &mut self,
        task: &mut SearchTask,
        task_sender: &TaskScheduler,
    ) -> Result<(), SelectError> {
        let SearchTask {
            num_visits,
            depth,
            ptr,
            nodepath: _,
            batch,
        } = task;

        let tree = &self.shared_tree.tree;
        let params = &self.shared_tree.params;
        let mut node = tree.get(*ptr);

        let is_root = *ptr == tree.root();
        if is_root && node.terminal_state != Terminal::Score {
            return Err(SelectError::TerminalRoot);
        }

        let mut children = node.children(tree);

        let fpu = params.calculate_fpu(node.calculate_q());

        // The children of the current node can be split into two categories
        // 1. Processed visits are visits to new nodes that have not been evaluated. These are either added to the batch or
        //    evaluated early if possible (i.e. terminal nodes or cached nodes).
        // 2. Scheduled visits are visits to existing nodes. If these nodes have already been evaluated by the neural network,
        //    then we will continue searching them. Otherwise, we can either visit it, if it has been evaluated, or count it
        //    as a collision
        let mut processed_visits = 0;
        let mut scheduled_visits = 0;

        let mut node_visits = node.all_visits();

        // Accumulators for merged backup of early evaluated nodes.
        let mut cumulative_early_wdl = WDL::default();
        let mut early_backprop_count = 0;

        for _ in 0..*num_visits {
            let best_child = select_best_child(&mut children, node_visits, fpu, params).unwrap();
            node_visits += 1;
            if best_child.is_uninstantiated() {
                let child_result = Searcher::process_uninstantiated_child(
                    &self.shared_tree,
                    batch,
                    &self.batch_stats,
                    &mut self.history,
                    *depth,
                    &mut node,
                    *ptr,
                    best_child,
                );

                processed_visits += 1;

                match child_result {
                    NodeStatus::Batch => (),
                    NodeStatus::Early(wdl) => {
                        cumulative_early_wdl += wdl;
                        early_backprop_count += 1;
                    }
                    NodeStatus::Terminal(terminal) => {
                        if is_root {
                            return Err(SelectError::TerminalRoot);
                        }
                        let parent = node.parent.unwrap();
                        node.virtual_visits += processed_visits;
                        node.terminal_state = terminal;

                        drop(node);
                        backprop_terminal(parent, terminal, tree);
                        // Since we need to return early, we have to cancel all visits except for those that have already been
                        // processed, i.e. added to the batch or backpropogated early
                        let total_scheduled_visits = *num_visits - processed_visits;

                        if !is_root {
                            cancel_visits(parent, total_scheduled_visits, tree);
                        }

                        // TODO: Maybe we don't want to backpropogate these. Although they are probably really rare...
                        execute_early_backprops(
                            *ptr,
                            early_backprop_count,
                            cumulative_early_wdl,
                            tree,
                        );

                        return Ok(());
                    }
                };
            } else {
                scheduled_visits += 1;
                best_child.scheduled_visits += 1;
            }
        }

        assert_eq!(processed_visits + scheduled_visits, *num_visits);
        assert_eq!(node_visits, node.all_visits() + (*num_visits as u32));

        node.virtual_visits += *num_visits;

        drop(node);

        execute_early_backprops(*ptr, early_backprop_count, cumulative_early_wdl, tree);

        if scheduled_visits > 0 {
            self.search_children(task, &children, task_sender)
        } else {
            Ok(())
        }
    }
}

fn select_best_child<'a>(
    children: &'a mut ChildrenList,
    parent_visits: u32,
    fpu: f32,
    params: &SearchParams,
) -> Option<&'a mut TempNode> {
    let mut best_puct_score = f32::NEG_INFINITY;

    let mut best_child = None;

    for child in children.iter_mut() {
        let value = match child.value {
            Some((score, terminal)) => match terminal {
                Terminal::Score => score,
                Terminal::Win(_) | Terminal::Loss(_) => continue,
                Terminal::Draw => 0.0,
            },
            // if the child is unevaluated, we fill in a default value called the FPU value or "First Play Urgency."
            // This value is calculated based on the parent value
            None => fpu,
        };

        let puct_score = calculate_puct(
            parent_visits,
            child.num_visits(),
            child.parent_edge,
            value,
            params.calculate_c(parent_visits),
        );

        assert!(puct_score > f32::NEG_INFINITY);

        if puct_score > best_puct_score {
            best_puct_score = puct_score;
            best_child = Some(child);
        }
    }

    best_child
}

fn calculate_puct(parent_visits: u32, child_visits: u32, edge: Edge, child_q: f32, c: f32) -> f32 {
    -child_q
        + (c * edge.get_policy() * ((parent_visits as f32).sqrt() / (child_visits as f32 + 1.0)))
}

enum NodeStatus {
    Batch,
    Early(WDL),
    Terminal(Terminal),
}

impl Searcher {
    #[allow(clippy::too_many_arguments)]
    fn process_uninstantiated_child(
        shared_tree: &SharedTreeData,
        batch: &Mutex<Batch>,
        batch_stats: &BatchStatistics,
        history: &mut PositionHistory,
        depth: u8,
        node: &mut NodeGuard,
        nodeptr: NodePtr,
        selected_child: &mut TempNode,
    ) -> NodeStatus {
        let tree = &shared_tree.tree;
        let tree_stats = &shared_tree.search_stats;

        // update statistics
        tree_stats.selective_depth.fetch_max(depth);
        tree_stats.cumulative_depth.fetch_add(depth as u64);
        tree_stats.node_count.fetch_add(1);

        let mut to_return = NodeStatus::Batch;

        history.make_move(selected_child.parent_edge.get_action());

        // Check if our current position can be evaluated without having to do an NN computation
        // There are two cases where this is possible:

        let (child_ptr, mut child_node) = tree
            .add_child(nodeptr, node, selected_child.parent_idx)
            .expect("Tree full!");

        child_node.virtual_visits += 1;

        selected_child.set_ptr(child_ptr);

        let terminal = history.terminal_type();
        // The current child is a terminal node, so we should check to make sure that our
        if terminal != Terminal::Score {
            child_node.terminal_state = terminal;
            child_node.solidify_visits(1);
            drop(child_node);
            match solve_terminal(node, terminal, tree) {
                Some(curr_terminal) => {
                    to_return = NodeStatus::Terminal(curr_terminal);
                }
                None => {
                    to_return = NodeStatus::Early(WDL::from(terminal));
                }
            }

            selected_child.value = Some((0.0, terminal));
        }
        // Alternatively, if it's in the cache (i.e. we've evaluated it already) we can also backprop
        else if let Some(entry) = shared_tree.cache.lock().probe(history.get_hash()) {
            child_node.set_nn_eval(&entry);
            child_node.solidify_visits(1);
            to_return = NodeStatus::Early(entry.wdl);

            selected_child.value = Some((entry.wdl.calculate_q(), Terminal::Score));
        }
        // Otherwise, if we can't evaluate it before sending it to the NN, add it to the batch for evaluation
        else {
            batch_stats.batch_count.fetch_add(1);
            batch
                .lock()
                .push_board(history, selected_child.ptr.unwrap());
        }

        selected_child.total_visits += 1;
        history.unmake_move();

        to_return
    }

    fn search_children(
        &mut self,
        task: &mut SearchTask,
        children: &ChildrenList,
        task_sender: &TaskScheduler,
    ) -> Result<(), SelectError> {
        let depth = task.depth;
        let num_visits = task.num_visits;
        let tree_data = &self.shared_tree;
        let tree = &tree_data.tree;
        let params = &tree_data.params;

        // Now, we have a list of all visits that are to be distributed across the children of the node. We will recursively call this function
        // on each of the children, but before then, let's partition off some of them to be taken care of by other threads.
        let mut local_tasks: ArrayVec<LocalTask, 255> = ArrayVec::new();

        for temp_child in children {
            let child_visits = temp_child.scheduled_visits;
            if child_visits == 0 {
                continue;
            }

            if temp_child.is_unevaluated() {
                // Child is already in the batch, pending evaluation. Therefore, we don't select further
                let mut child_node = tree.get(temp_child.ptr.unwrap());
                child_node.virtual_visits += child_visits;
                self.batch_stats.num_collisions.fetch_add(child_visits);
            } else if temp_child.terminal_state().unwrap() != Terminal::Score {
                // Note that we don't backpropogate terminal because we were unable to solve the terminality of the parent.
                // In most cases, this means that it's a draw, and we want to weight it with as many visits as were scheduled
                // for it.
                backprop_eval(
                    task.ptr,
                    WDL::from(temp_child.terminal_state().unwrap()),
                    child_visits,
                    tree,
                );
            } else if params.should_dispatch(num_visits, temp_child.scheduled_visits) {
                // In certain cases, we want to schedule a node for a helper thread
                task.nodepath.push(temp_child.parent_edge.get_action());
                task_sender.send_task(SearchTask {
                    num_visits: child_visits,
                    depth: depth + 1,
                    ptr: temp_child.ptr.unwrap(),
                    nodepath: task.nodepath.clone(),
                    batch: Arc::clone(&task.batch),
                });
                task.nodepath.pop();
            } else {
                local_tasks.push(LocalTask::from(temp_child))
            }
        }

        // Now, handle the tasks that can be applied locally
        task.depth += 1;
        for local_task in local_tasks {
            self.history.make_move(local_task.action);
            task.nodepath.push(local_task.action);
            task.ptr = local_task.ptr;
            task.num_visits = local_task.num_visits;

            match self.multi_select(task, task_sender) {
                Ok(()) => (),
                Err(e) => {
                    self.history.unmake_move();
                    task.nodepath.pop();
                    return Err(e);
                }
            };
            task.nodepath.pop();
            self.history.unmake_move();
        }
        task.depth -= 1;

        Ok(())
    }
}

struct LocalTask {
    num_visits: u16,
    ptr: NodePtr,
    action: Action,
}

impl From<&TempNode> for LocalTask {
    fn from(value: &TempNode) -> Self {
        Self {
            num_visits: value.scheduled_visits,
            ptr: value.ptr.unwrap(),
            action: value.parent_edge.get_action(),
        }
    }
}

fn execute_early_backprops(
    nodeptr: NodePtr,
    early_backprop_count: u16,
    cumulative_early_wdl: WDL,
    tree: &NodeTree,
) {
    if early_backprop_count > 0 {
        let wdl = cumulative_early_wdl / (early_backprop_count as f32);
        backprop_eval(nodeptr, wdl, early_backprop_count, tree);
    }
}

fn cancel_visits(ptr: NodePtr, num_visits: u16, tree: &NodeTree) {
    let mut node = tree.get(ptr);
    assert!(node.virtual_visits >= num_visits);
    node.virtual_visits -= num_visits;
    if ptr != tree.root() {
        let parent = node.parent.unwrap();
        drop(node);
        cancel_visits(parent, num_visits, tree)
    }
}

pub fn process_backprop_request(task: &BackpropData, tree: &NodeTree) {
    let entry = task.entry;
    let mut child_node = tree.get(entry.ptr.unwrap());

    // when we have collisions, one network evaluation is treated like multiple visits. In other words, we weight
    // visits by the number of collisions, despite the fact that they gain no new information. This is to avoid
    // value dilution, which is when the influence of weaker nodes unnecessarily drags down the value of their parents
    let num_visits = child_node.virtual_visits;
    child_node.set_nn_eval(&task.eval);
    child_node.solidify_visits(num_visits);
    let parent = child_node.parent.unwrap();
    drop(child_node);
    backprop_eval(parent, task.eval.wdl, num_visits, tree);
}

fn backprop_eval(nodeptr: NodePtr, child_eval: WDL, visit_count: u16, tree: &NodeTree) {
    assert!(visit_count != 0);
    let mut node = tree.get(nodeptr);
    let parent_wdl = child_eval.parent_backup_value();

    let new_wdl = weighted_avg(
        parent_wdl,
        visit_count as f32,
        node.wdl,
        node.full_visits as f32,
    );

    node.wdl = new_wdl;

    node.solidify_visits(visit_count);

    if nodeptr != tree.root() {
        let parent = node.parent.unwrap();
        drop(node);
        backprop_eval(parent, parent_wdl, visit_count, tree)
    }
}

fn backprop_terminal(nodeptr: NodePtr, child_terminal: Terminal, tree: &NodeTree) {
    let mut node = tree.get(nodeptr);
    match solve_terminal(&node, child_terminal, tree) {
        Some(node_terminal) => {
            node.terminal_state = node_terminal;
            node.solidify_visits(1);
            if nodeptr != tree.root() {
                let parent = node.parent.unwrap();
                drop(node);
                backprop_terminal(parent, node_terminal, tree)
            }
        }
        None => {
            drop(node);
            backprop_eval(nodeptr, WDL::from(child_terminal), 1, tree)
        }
    }
}

fn solve_terminal(node: &NodeGuard, child_terminal: Terminal, tree: &NodeTree) -> Option<Terminal> {
    let update_state = match child_terminal {
        Terminal::Score => panic!("Why are we trying to backpropogate this?"),
        Terminal::Win(_) | Terminal::Draw => {
            // If all of our children are wins (from the child's perspectiv), we are lost
            // if all of our children are drawn or won, we are drawn.
            let mut to_propogate = child_terminal;
            let mut max_ply = 0;
            for child in node.children(tree) {
                match child.terminal_state()? {
                    Terminal::Score | Terminal::Loss(_) => return None,
                    Terminal::Win(plies) => {
                        if plies > max_ply {
                            max_ply = plies
                        }
                    }
                    Terminal::Draw => to_propogate = Terminal::Draw,
                }
            }
            if matches!(to_propogate, Terminal::Loss(_)) {
                Terminal::Loss(max_ply)
            } else {
                to_propogate
            }
        }
        Terminal::Loss(ply) => {
            if let Terminal::Win(best_ply) = node.terminal_state {
                if ply > best_ply {
                    return Some(Terminal::Win(best_ply));
                }
            }
            Terminal::Win(ply) // we have one losing child, so we are winning
        }
    };
    Some(update_state)
}
