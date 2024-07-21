use anyhow::Context;
use ort::{
    ExecutionProvider, GraphOptimizationLevel, OutputSelector, RunOptions, Session, Tensor,
    TensorRTExecutionProvider,
};
use parking_lot::Mutex;

use super::{batch::Batch, network::Network};

pub struct OnnxTRTNetwork {
    session: Mutex<Session>,
}

use crate::mcts::params::{DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, MIN_BATCH_SIZE};

impl OnnxTRTNetwork {
    pub fn new(filepath: &str, device_idx: usize) -> anyhow::Result<Self> {
        let tensorrt_ep = TensorRTExecutionProvider::default()
            .with_device_id(device_idx as i32)
            .with_engine_cache(true)
            .with_engine_cache_path("trt_engines")
            .with_timing_cache(true)
            .with_fp16(true)
            .with_build_heuristics(true)
            .with_sparsity(true)
            .with_builder_optimization_level(1)
            .with_profile_min_shapes(format!("/input/planes:{MIN_BATCH_SIZE}x112x8x8"))
            .with_profile_max_shapes(format!("/input/planes:{MAX_BATCH_SIZE}x112x8x8"))
            .with_profile_opt_shapes(format!("/input/planes:{DEFAULT_BATCH_SIZE}x112x8x8"));

        let builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Disable)?;

        tensorrt_ep.register(&builder).unwrap();

        let session = builder.commit_from_file(filepath)?;

        Self::verify(&session)
            .context("Invalid input and output dimensions! Failed to load model!")?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    fn verify(session: &Session) -> Option<()> {
        if session.inputs.len() == 1
            && session.inputs[0].input_type.tensor_dimensions()? == &[-1, 112, 8, 8]
            && session.outputs.len() == 3
            && session.outputs[0].output_type.tensor_dimensions()? == &[-1, 1858]
            && session.outputs[1].output_type.tensor_dimensions()? == &[-1, 3]
            && session.outputs[2].output_type.tensor_dimensions()? == &[-1, 1]
        {
            Some(())
        } else {
            None
        }
    }
}

impl Network for OnnxTRTNetwork {
    fn locked_compute(&self, batch: &mut Batch) -> anyhow::Result<()> {
        let input_tensor = Tensor::from_array(batch.input_tensor.view())?;
        let output_policy = Tensor::from_array(&mut batch.output_policy)?;
        let output_wdl = Tensor::from_array(&mut batch.output_wdl)?;
        let output_mlh = Tensor::from_array(&mut batch.output_mlh)?;

        let session = &self.session.lock();
        let policy_name = session.outputs[0].name.as_str();
        let wdl_name = session.outputs[1].name.as_str();
        let mlh_name = session.outputs[2].name.as_str();
        let options = RunOptions::new()?.with_outputs(
            OutputSelector::no_default()
                .with(policy_name)
                .with(wdl_name)
                .with(mlh_name)
                .preallocate(policy_name, output_policy)
                .preallocate(wdl_name, output_wdl)
                .preallocate(mlh_name, output_mlh),
        );

        session.run_with_options(ort::inputs![input_tensor]?, &options)?;

        Ok(())
    }

    fn is_unlocked(&self) -> bool {
        self.session.try_lock().is_some()
    }
}
