use super::batch::Batch;

pub trait Network: Send + Sync + 'static {
    fn locked_compute(&self, batch: &mut Batch) -> anyhow::Result<()>;
    fn is_unlocked(&self) -> bool;
}
