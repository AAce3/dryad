use std::f32::consts::LN_2;

use fast_math::log2_raw;


use super::search_driver::BatchStatistics;


pub const MIN_BATCH_SIZE: usize = 1;
pub const MAX_BATCH_SIZE: usize = 2048;
pub const DEFAULT_BATCH_SIZE: usize = 256;

#[derive(Clone)]
pub struct SearchParams {
    pub(super) cpuct_init: f32,
    pub(super) cpuct_base: f32,
    pub(super) cpuct_weight: f32,
    pub(super) fpu_reduction: f32,
    pub policy_temperature: f32,
    pub(super) min_task_size: u16,
    pub(super) _draw_value: f32,
    pub batch_size: u16,
    pub num_workers: u16,
    pub(super) max_collisions: u16,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            cpuct_init: 2.00,
            cpuct_base: 30_000.0,
            cpuct_weight: 4.0,
            fpu_reduction: 0.5,
            policy_temperature: 1.4,
            min_task_size: 16,
            _draw_value: 0.0,
            batch_size: DEFAULT_BATCH_SIZE as u16,
            num_workers: 0,
            max_collisions: 1024,
        }
    }
}

impl SearchParams {
    // PUCT equation to calculate C:
    //
    //                                       (parent_visits + cpuct_base + 1)
    // C = cpuct_init + cpuct_weight *  log  ----------------------------------
    //                                               cpuct_base
    //
    // We want to prioritize exploitation at low parent node counts to avoid exploring too many bad nodes
    // At high node counts, we have the luxury of exploring more
    // growth weight and  controls how much the C-value grows with parent visits

    // PUCT formula to get next node:
    //
    //                            sqrt(parent_N)
    // child_Q + C * child_P * ---------------------
    //                              1 + child_N
    //

    pub fn calculate_c(&self, parent_visits: u32) -> f32 {
        let delta = self.cpuct_weight
            * LN_2
            * log2_raw(((parent_visits as f32) + self.cpuct_base + 1.0) / self.cpuct_base);
        self.cpuct_init + delta
    }

    pub fn should_dispatch(&self, parent_visits: u16, child_visits: u16) -> bool {
        self.num_workers > 0
            && child_visits > self.min_task_size
            && ((child_visits as f32) / (parent_visits as f32) < 0.5)
    }

    pub fn calculate_fpu(&self, parent_value: f32) -> f32 {
        -(parent_value - (self.fpu_reduction))
    }

    pub fn next_search_size(&self, batch_stats: &BatchStatistics) -> u16 {
        self.selection_size(
            batch_stats.batch_count.load(),
            batch_stats.num_collisions.load(),
        )
    }

    pub fn selection_size(&self, batch_count: u16, num_collisions: u16) -> u16 {
        u16::min(
            self.batch_size - batch_count,
            self.max_collisions - num_collisions,
        )
    }

    pub fn default_selection_size(&self) -> u16 {
        self.selection_size(0, 0)
    }
}

pub struct TreeParams {
    pub max_tree_size: usize,
    pub cache_size: usize,
    pub move_overhead_ms: u64,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_tree_size: 5_000_000,
            cache_size: 2_000_000,
            move_overhead_ms: 10
        }
    }
}
