//! Point to point communication traits
pub use super::{Source, Destination, Send, BufferedSend, SynchronousSend, ReadySend, Probe,
                MatchedReceive, MatchedReceiveInto, MatchedReceiveVec, MatchedProbe,
                Receive, ReceiveInto, ReceiveVec,
                SendReceive, SendReceiveInto, SendReceiveReplaceInto,
                Wait, Test,
                ImmediateSend, ImmediateBufferedSend, ImmediateSynchronousSend, ImmediateReadySend,
                ImmediateReceive, ImmediateReceiveInto, ImmediateProbe, ImmediateMatchedProbe,
                ImmediateMatchedReceiveInto};
