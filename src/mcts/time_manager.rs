use std::time::Instant;

use super::search_driver::SearchStatistics;

pub struct TimeManager {
    pub allocated_time_ms: Option<u64>,
    pub max_nodes: Option<u32>,
    pub max_depth: Option<u8>,
    timer: Instant,
}

impl TimeManager {
    pub fn new(
        time_ms: Option<u64>,
        increment_ms: Option<u64>,
        move_time: Option<u64>,
        max_nodes: Option<u32>,
        max_depth: Option<u8>,
    ) -> Self {
        let tournament_time = time_ms.map(|a| (a / 20) + increment_ms.unwrap_or(0) / 2);

        Self {
            allocated_time_ms: tournament_time.or(move_time),
            max_nodes,
            max_depth,
            timer: Instant::now(),
        }
    }

    pub fn should_stop(&self, search_stats: &SearchStatistics) -> bool {
        let mut should_stop = false;

        if let Some(allocated_time) = self.allocated_time_ms {
            should_stop |= self.get_time_millis() > allocated_time
        }

        if let Some(max_nodes) = self.max_nodes {
            should_stop |= search_stats.node_count.load() > max_nodes
        }

        if let Some(max_depth) = self.max_depth {
            should_stop |= search_stats.avg_depth() > max_depth
        }

        should_stop
    }

    pub fn get_time_millis(&self) -> u64 {
        self.timer.elapsed().as_millis() as u64
    }

    pub fn get_nps(&self, nodes: u32) -> f32 {
        nodes as f32 / self.timer.elapsed().as_secs_f32()
    }
}
