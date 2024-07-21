// use anyhow::Context;
// use kn_cuda_eval::{executor::CudaExecutor, CudaDevice};
// use kn_graph::{
//     dtype::DTensor,
//     onnx::{load_graph_from_onnx_path, GraphLoader},
//     optimizer::{optimize_graph, OptimizerSettings},
//     shape::{Shape, Size},
//     shape
// };

// use super::{batch::Batch, network::Network};

// pub struct KnNet {
//     executor: CudaExecutor,
// }

// impl Network for KnNet {
//     fn new(filepath: &str, device_idx: usize, batch_size: usize) -> anyhow::Result<Self> {
//         let mut loader = GraphLoader::from_path(filepath, false)?;
//         loader.force_input_shapes(vec![Some(Shape::new(vec![Size::BATCH, 112.into(), 8.into(), 8.into()]))]);
//         let graph = loader.load()?;
//         let graph = optimize_graph(&graph, OptimizerSettings::default());
//         let device = CudaDevice::new(device_idx as i32).unwrap();
//         let executor = CudaExecutor::new(device, &graph, batch_size);

//         Ok(Self { executor })
//     }

//     fn compute(&mut self, batch: &mut Batch) -> anyhow::Result<()> {
//         let input = [DTensor::F32(batch.input_tensor.clone().into_dyn())];
//         let results = self.executor.evaluate(&input);
//         let policy = results[0].unwrap_f32().context("Failed to evaluate")?;
//         let wdl = results[1].unwrap_f32().context("Failed to evaluate!")?;
//         let mlh = results[2].unwrap_f32().context("Failed to evaluate!")?;

//         batch.output_policy = policy
//             .clone()
//             .into_shape((batch.batch_size, Batch::POLICY_SIZE))?;
//         batch.output_wdl = wdl
//             .clone()
//             .into_shape((batch.batch_size, Batch::WDL_SIZE))?;
//         batch.output_mlh = mlh
//             .clone()
//             .into_shape((batch.batch_size, Batch::MLH_SIZE))?;

//         Ok(())
//     }
// }
