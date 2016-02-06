//! Collective communication traits
pub use super::{Barrier, Root, BroadcastInto, GatherInto, GatherVarcountInto,
                AllGatherInto, AllGatherVarcountInto, ScatterInto, ScatterVarcountInto,
                AllToAllInto, Operation, ReduceInto, AllReduceInto, ScanInto, ExclusiveScanInto,
                ImmediateBarrier, ImmediateBroadcastInto, ImmediateGatherInto, ImmediateAllGatherInto,
                ImmediateScatterInto, ImmediateAllToAllInto};
