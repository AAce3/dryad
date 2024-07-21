use std::
    num::NonZeroU32
;

use parking_lot::{Mutex, MutexGuard};

use super::node::Node;

pub type NodeLock = Mutex<Node>;
pub type NodeGuard<'a> = MutexGuard<'a, Node>;



pub type NodePtr = ArenaPtr;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct ArenaPtr {
    idx: NonZeroU32,
}

impl ArenaPtr{
    pub fn idx(&self) -> usize {
        u32::from(self.idx) as usize
    }
}

impl From<u32> for ArenaPtr {
    fn from(value: u32) -> Self {
        Self {
            idx: NonZeroU32::new(value).unwrap(),
        }
    }
}

impl From<NonZeroU32> for ArenaPtr {
    fn from(value: NonZeroU32) -> Self {
        Self { idx: value }
    }
}
