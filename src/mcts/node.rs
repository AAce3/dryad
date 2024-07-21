use std::mem;

use arrayvec::ArrayVec;
use auto_ops::impl_op;

use crate::movegen::action::Action;

use super::cache::NNEval;
use super::ptrs::NodePtr;
use super::tree::NodeTree;

// Node storage is exactly the same as Lc0 (non-DAG). Each node stores a linked list of its children, as well as the parent index.
// However, the nodes are stored in a large slotmap, with 32-bit indices as pointers
//
// Example config (from Lc0 documentation):
//
//                                Parent Node
//                                    |
//        +-------------+-------------+----------------+--------------+
//        |              |            |                |              |
//   Edge 0(Nf3)    Edge 1(Bc5)     Edge 2(a4)     Edge 3(Qxf7)    Edge 4(a3)
//    (dangling)         |           (dangling)        |           (dangling)
//                   Node, Q=0.5                    Node, Q=-0.2
//
//  Is represented as:
// +--------------+
// | Parent Node  |
// +--------------+                                        +--------+
// | edges        | -------------------------------------> | Edges  |
// |              |    +------------+                      +--------+
// | first_child  | -> | Node       |                      | Nf3    |
// +--------------+    +------------+                      | Bc5    |
//                     | idx = 1    |                      | a4     |
//                     | wdl = 0.5  |    +------------+    | Qxf7   |
//                     | sibling    | -> | Node       |    | a3     |
//                     +------------+    +------------+    +--------+
//                                       | idx = 3    |
//                                       | wdl = -0.2 |
//                                       | sibling    | -> None
//                                       +------------+

#[derive(Default)]
pub struct Node {
    pub wdl: WDL,
    pub full_visits: u32,
    pub virtual_visits: u16,
    pub edges: Box<[Edge]>,
    pub first_child: Option<NodePtr>,
    pub sibling: Option<NodePtr>,
    pub parent: Option<NodePtr>,
    pub parent_edge_idx: u8,
    pub terminal_state: Terminal,
}

impl Node {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn all_visits(&self) -> u32 {
        self.full_visits + (self.virtual_visits as u32)
    }

    pub fn calculate_q(&self) -> f32 {
        self.wdl.calculate_q()
    }

    pub fn set_nn_eval(&mut self, nn_eval: &NNEval) {
        self.wdl = nn_eval.wdl;
        self.edges = Box::from_iter(nn_eval.edges.iter().copied());
    }

    pub fn is_solid(&self) -> bool {
        self.full_visits != 0
    }

    pub fn children(&self, tree: &NodeTree) -> ChildrenList {
        let edges = self.edges.as_ref();

        let mut children: ArrayVec<TempNode, 255> = ArrayVec::from_iter(
            edges
                .iter()
                .enumerate()
                .map(|(idx, edge)| TempNode::from_edge(edge, idx)),
        );

        let mut child_ptr = self.first_child;

        while let Some(ptr) = child_ptr {
            let child = tree.get(ptr);
            let idx = child.parent_edge_idx as usize;
            child.copy_to_tempnode(child_ptr.unwrap(), &mut children[idx]);
            child_ptr = child.sibling;
        }

        children
    }

    pub fn copy_to_tempnode(&self, own_ptr: NodePtr, temp_node: &mut TempNode) {
        let value = if self.is_solid() {
            Some((self.calculate_q(), self.terminal_state))
        } else {
            None
        };
        let parent_edge = temp_node.parent_edge;
        *temp_node = TempNode {
            value,
            total_visits: self.full_visits + (self.virtual_visits as u32),
            scheduled_visits: 0,
            ptr: Some(own_ptr),
            parent_edge,
            parent_idx: self.parent_edge_idx as usize,
        }
    }

    pub fn solidify_visits(&mut self, num_visits: u16) {
        assert!(
            self.virtual_visits >= num_visits,
            "{} < {}",
            self.virtual_visits,
            num_visits
        );
        self.virtual_visits -= num_visits;
        self.full_visits += num_visits as u32
    }
}

#[derive(Default)]
pub struct TempNode {
    pub value: Option<(f32, Terminal)>,
    pub total_visits: u32,
    pub scheduled_visits: u16,
    pub ptr: Option<NodePtr>,
    pub parent_edge: Edge,
    pub parent_idx: usize,
}

pub type ChildrenList = ArrayVec<TempNode, 255>;

impl TempNode {
    pub fn from_edge(edge: &Edge, parent_idx: usize) -> Self {
        Self {
            parent_edge: *edge,
            parent_idx,
            ..Default::default()
        }
    }

    pub fn terminal_state(&self) -> Option<Terminal> {
        self.value.map(|a| a.1)
    }

    pub fn q_value(&self) -> Option<f32> {
        self.value.map(|a| a.0)
    }

    pub fn num_visits(&self) -> u32 {
        self.total_visits + (self.scheduled_visits as u32)
    }

    pub fn is_unevaluated(&self) -> bool {
        self.value.is_none()
    }

    pub fn is_uninstantiated(&self) -> bool {
        self.ptr.is_none()
    }

    pub fn set_ptr(&mut self, new_ptr: NodePtr) {
        assert!(
            self.is_uninstantiated(),
            "Don't try to change the ptr of an existing node"
        );
        self.ptr = Some(new_ptr)
    }
}

// specialized node used only for selection

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Edge {
    policy: u16,
    action: Action,
}

impl Edge {
    // lc0 implementation of packing 32 bit floats into 16 bit integers
    pub fn new(policy: f32, action: Action) -> Self {
        assert!((0.0..=1.0).contains(&policy));
        const ROUNDINGS: i32 = (1 << 11) - (3 << 28);
        unsafe {
            let temp = mem::transmute_copy::<f32, i32>(&policy) + ROUNDINGS;
            let policy_packed = if temp < 0 { 0 } else { (temp >> 12) as u16 };
            Edge {
                policy: policy_packed,
                action,
            }
        }
    }

    pub fn get_policy(&self) -> f32 {
        let temp = ((self.policy as u32) << 12) | (3 << 28);
        f32::from_bits(temp)
    }

    pub fn get_action(&self) -> Action {
        self.action
    }
}

#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct WDL {
    pub win_loss: f32,
    pub draw: f32,
    pub moves_left: f32,
}

impl WDL {
    pub const WIN: Self = WDL {
        win_loss: 1.0,
        draw: 0.0,
        moves_left: 0.0,
    };
    pub const LOSS: Self = WDL {
        win_loss: -1.0,
        draw: 0.0,
        moves_left: 0.0,
    };
    pub const DRAW: Self = WDL {
        win_loss: 0.0,
        draw: 1.0,
        moves_left: 0.0,
    };

    pub fn calculate_q(&self) -> f32 {
        self.win_loss
    }

    pub fn parent_backup_value(&self) -> WDL {
        WDL {
            win_loss: -self.win_loss,
            draw: self.draw,
            moves_left: self.moves_left + 1.0,
        }
    }
}

pub fn weighted_avg(wdl_1: WDL, weight_1: f32, wdl_2: WDL, weight_2: f32) -> WDL {
    (wdl_1 * weight_1 + wdl_2 * weight_2) / (weight_1 + weight_2)
}

impl_op!(+ |a: WDL, b: WDL| -> WDL {
    WDL {
        win_loss: a.win_loss + b.win_loss,
        draw: a.draw + b.draw,
        moves_left: a.moves_left + b.moves_left
    }
});

impl_op!(+= |a: &mut WDL, b: WDL| {
    a.win_loss += b.win_loss;
    a.draw += b.draw;
    a.moves_left += b.moves_left
});

impl_op!(*|a: WDL, b: f32| -> WDL {
    WDL {
        win_loss: a.win_loss * b,
        draw: a.draw * b,
        moves_left: a.moves_left * b,
    }
});

impl_op!(*= |a: &mut WDL, b: f32|{
    a.win_loss *= b;
    a.draw *= b;
    a.moves_left *= b;
});

impl_op!(/ |a: WDL, b: f32| -> WDL {
    WDL {
        win_loss: a.win_loss / b ,
        draw: a.draw / b, moves_left:
        a.moves_left / b
    }
});

impl_op!(/= |a: &mut WDL, b: f32|{
    a.win_loss /= b;
    a.draw /= b;
    a.moves_left /= b;
});

impl_op!(-|a: WDL| -> WDL {
    WDL {
        win_loss: -a.win_loss,
        draw: a.draw,
        moves_left: a.moves_left,
    }
});

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Terminal {
    #[default]
    Score,
    Win(u8),
    Loss(u8),
    Draw,
}

impl From<Terminal> for WDL {
    fn from(val: Terminal) -> Self {
        match val {
            Terminal::Score => panic!("Don't try to figure out the score right now"),
            Terminal::Win(plies) => {
                let mut wdl = WDL::WIN;
                wdl.moves_left = plies as f32;
                wdl
            }
            Terminal::Loss(plies) => {
                let mut wdl = WDL::LOSS;
                wdl.moves_left = plies as f32;
                wdl
            }
            Terminal::Draw => WDL::DRAW,
        }
    }
}
