//! Collective communication traits
pub use super::{Barrier, Root, BroadcastInto, GatherInto, AllGatherInto, ScatterInto,
                AllToAllInto, ReduceInto, AllReduceInto, ScanInto, ExclusiveScanInto,
                ImmediateBarrier, ImmediateBroadcastInto, ImmediateGatherInto, ImmediateAllGatherInto,
                ImmediateScatterInto, ImmediateAllToAllInto};
