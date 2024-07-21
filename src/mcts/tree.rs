use crossbeam::atomic::AtomicCell;
use parking_lot::Mutex;

use crate::movegen::history::PositionHistory;

use super::{
    node::{Edge, Node},
    ptrs::{NodeGuard, NodeLock, NodePtr},
};

// A slotmap for nodes
// Free nodes are stored as a linked list of indices.
// When allocating a node, find the next free node and return it.
pub struct NodeArena {
    arena: Vec<NodeLock>,
    free_nodes: Mutex<Vec<NodePtr>>,
}

impl<'a> NodeArena {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity + 1;
        let mut arena = Vec::with_capacity(capacity + 1);
        let mut free = Vec::with_capacity(capacity + 1);
        for i in (0..(capacity + 1)).rev() {
            arena.push(NodeLock::new(Node::default()));
            if i != 0 {
                free.push(NodePtr::from(i as u32));
            }
        }
        assert!(arena.len() == capacity + 1);
        Self {
            arena,
            free_nodes: Mutex::new(free),
        }
    }

    pub fn reset(&self) {
        let mut free = self.free_nodes.lock();
        free.clear();
        for i in (0..self.arena.len()).rev() {
            self.arena[i].lock().clear();
            if i != 0 {
                free.push(NodePtr::from(i as u32));
            }
        }
    }

    pub fn get(&'a self, ptr: NodePtr) -> NodeGuard {
        self.arena[ptr.idx()].lock()
    }

    pub fn allocate(&self) -> Option<NodePtr> {
        self.free_nodes.lock().pop()
    }

    pub fn remaining_nodes(&self) -> u32 {
        self.free_nodes.lock().len() as u32
    }

    pub fn deallocate(&self, ptr: NodePtr) {
        self.deallocate_with_lock(ptr, self.get(ptr))
    }

    pub fn deallocate_with_lock(&self, ptr: NodePtr, mut node: NodeGuard) {
        node.clear();
        self.free_nodes.lock().push(ptr);
    }
}

pub struct NodeTree {
    pub arena: NodeArena,
    root: AtomicCell<Option<NodePtr>>,
    pub node_count: AtomicCell<u32>,
}

impl NodeTree {
    pub fn new(capacity: usize) -> Self {
        Self {
            arena: NodeArena::new(capacity),
            root: AtomicCell::new(None),
            node_count: AtomicCell::new(0),
        }
    }

    pub fn add_child(
        &self,
        parent_ptr: NodePtr,
        parent: &mut NodeGuard,
        edge_idx: usize,
    ) -> Option<(NodePtr, NodeGuard)> {
        let (child_ptr, mut child_node) = self.allocate_node()?;

        child_node.parent = Some(parent_ptr);
        child_node.sibling = parent.first_child;
        parent.first_child = Some(child_ptr);

        child_node.parent_edge_idx = edge_idx as u8;

        Some((child_ptr, child_node))
    }

    pub fn try_prune(
        &self,
        current_board: &mut PositionHistory,
        target_board: &mut PositionHistory,
        max_depth: u8,
    ) -> Option<NodePtr> {
        let new_root =
            self.recurse_find(self.root.load()?, current_board, target_board, max_depth)?;
        self.prune_except_new_root(self.root(), new_root);
        self.set_root(new_root);
        Some(new_root)
    }

    // Try to prune the tree until the
    // logic taken from https://github.com/official-monty/Monty
    fn recurse_find(
        &self,
        ptr: NodePtr,
        current_board: &mut PositionHistory,
        target_board: &mut PositionHistory,
        depth: u8,
    ) -> Option<NodePtr> {
        if current_board.latest_pos() == target_board.latest_pos()
            && current_board.num_repetitions() == target_board.num_repetitions()
        {
            return Some(ptr);
        }

        if depth == 0 {
            return None;
        }

        let node = self.get(ptr);

        for child in node.children(self) {
            if let Some(child_ptr) = child.ptr {
                current_board.make_move(child.parent_edge.get_action());
                let result = self.recurse_find(child_ptr, current_board, target_board, depth - 1);
                current_board.unmake_move();
                if result.is_some() {
                    return result;
                }
            }
        }

        None
    }

    // prunes node and all of its children, preserving the new root
    fn prune_except_new_root(&self, nodeptr: NodePtr, new_root: NodePtr) {
        let node = self.get(nodeptr);
        for child in node.children(self) {
            if let Some(ptr) = child.ptr {
                if ptr == new_root {
                    continue;
                } else {
                    self.prune_except_new_root(ptr, new_root);
                }
            }
        }
        self.deallocate_with_lock(nodeptr, node);
    }

    pub fn allocate_node(&self) -> Option<(NodePtr, NodeGuard)> {
        let child_ptr = self.arena.allocate()?;

        let child_node = self.get(child_ptr);

        self.node_count.fetch_add(1);

        Some((child_ptr, child_node))
    }

    pub fn deallocate_node(&self, ptr: NodePtr) {
        self.arena.deallocate(ptr);
        self.node_count.fetch_sub(1);
    }

    fn deallocate_with_lock(&self, ptr: NodePtr, node: NodeGuard) {
        self.arena.deallocate_with_lock(ptr, node);
        self.node_count.fetch_sub(1);
    }

    pub fn root(&self) -> NodePtr {
        self.root.load().unwrap()
    }

    pub fn set_root(&self, ptr: NodePtr) {
        self.root.store(Some(ptr))
    }

    pub fn get(&self, ptr: NodePtr) -> NodeGuard {
        self.arena.get(ptr)
    }

    pub fn clear(&self) {
        self.arena.reset();
        self.node_count.store(0);
    }

    pub fn remaining_nodes(&self) -> u32 {
        self.arena.remaining_nodes()
    }
}

impl NodeTree {
    pub fn to_graphviz(&self) -> String {
        let mut result = String::from("digraph Tree {\n");
        result.push_str("    node [shape=record];\n");

        if let Some(root) = self.root.load() {
            self.node_to_graphviz(0, root, None, &mut result);
        }

        result.push_str("}\n");
        result
    }

    fn node_to_graphviz(
        &self,
        depth: u8,
        node_ptr: NodePtr,
        edge: Option<Edge>,
        result: &mut String,
    ) {
        let node_info;
        let children_info;

        {
            let node = self.get(node_ptr);

            node_info = (
                node_ptr.idx(),
                if depth % 2 == 0 {
                    node.calculate_q()
                } else {
                    -node.calculate_q()
                },
                node.full_visits,
                node.virtual_visits,
                node.first_child,
                node.edges.clone(),
            );

            children_info = node
                .first_child
                .map(|first_child| {
                    let mut children = Vec::new();
                    let mut child_ptr = Some(first_child);
                    while let Some(ptr) = child_ptr {
                        let child = self.get(ptr);
                        children.push((ptr, child.parent_edge_idx, child.sibling));
                        child_ptr = child.sibling;
                    }
                    children
                })
                .unwrap_or_default();
        }

        let label = if let Some(edge) = edge {
            format!(
                "{} | {{ Q: {:.3} | Visits: {} | V-Visits: {} | Move: {}}}",
                node_info.0,
                node_info.1,
                node_info.2,
                node_info.3,
                edge.get_action()
            )
        } else {
            format!(
                "{} | {{ Q: {:.3} | Visits: {} | V-Visits: {}}}",
                node_info.0, node_info.1, node_info.2, node_info.3
            )
        };

        result.push_str(&format!("    {} [label=\"{}\"];\n", node_info.0, label));

        for (child_ptr, parent_edge_idx, _) in children_info {
            let edge = &node_info.5[parent_edge_idx as usize];

            let edge_label = format!(
                "Action: {} | Policy: {:.3}",
                edge.get_action(),
                edge.get_policy()
            );

            result.push_str(&format!(
                "    {} -> {} [label=\"{}\"];\n",
                node_info.0,
                child_ptr.idx(),
                edge_label
            ));

            self.node_to_graphviz(depth + 1, child_ptr, Some(*edge), result);
        }
    }
}
