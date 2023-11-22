// HashMap Benchmark for upstream NR
// Adapted from https://github.com/vmware/node-replication/blob/master/node-replication/benches/hashmap/main.rs
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Defines a hash-map that can be replicated.
#![allow(dead_code)]
use std::fmt::Debug;
use std::marker::Sync;
use std::num::NonZeroUsize;
use std::sync::Arc;

use logging::warn;
use rand::seq::SliceRandom;

use bench_utils::benchmark::*;
use bench_utils::mkbench::{self, DsInterface};
use bench_utils::topology::ThreadMapping;
use bench_utils::Operation;


#[cfg(feature="standard")]
use node_replication::nr::{Dispatch, NodeReplicated, ReplicaId, ThreadToken};

#[cfg(feature="verified")]
use verus_nr::{Dispatch, NodeReplicated, ReplicaId};


/// The initial amount of entries all Hashmaps are initialized with
#[cfg(feature = "smokebench")]
pub const INITIAL_CAPACITY: usize = 1 << 22; // ~ 4M
#[cfg(not(feature = "smokebench"))]
pub const INITIAL_CAPACITY: usize = 1 << 26; // ~ 67M

// Biggest key in the hash-map
#[cfg(feature = "smokebench")]
pub const KEY_SPACE: usize = 5_000_000;
#[cfg(not(feature = "smokebench"))]
pub const KEY_SPACE: usize = 50_000_000;

// Key distribution for all hash-maps [uniform|skewed]
pub const UNIFORM: &'static str = "uniform";

// Number of operation for test-harness.
#[cfg(feature = "smokebench")]
pub const NOP: usize = 2_500_000;
#[cfg(not(feature = "smokebench"))]
pub const NOP: usize = 25_000_000;

/// Operations we can perform on the stack.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum OpWr {
    /// Increment the Counter
    Inc
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum OpRd {
    /// Get the counter value
    Get,
}

/// Single-threaded implementation of the stack
///
/// We just use a vector.
#[derive(Debug, Clone)]
pub struct NrCounter {
    counter: u64,
}

impl NrCounter {
    pub fn inc(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }
    pub fn get(&self) -> u64 {
        self.counter
    }
}

impl Default for NrCounter {
    /// Return a dummy hash-map with `INITIAL_CAPACITY` elements.
    fn default() -> NrCounter {
        NrCounter { counter: 0 }
    }
}

impl Dispatch for NrCounter {
    type ReadOperation<'rop> = OpRd;
    type WriteOperation = OpWr;
    type Response = Result<u64, ()>;

    fn dispatch<'rop>(&self, op: Self::ReadOperation<'rop>) -> Self::Response {
        match op {
            OpRd::Get => return Ok(self.get()),
        }
    }

    /// Implements how we execute operation from the log against our local stack
    fn dispatch_mut(&mut self, op: Self::WriteOperation) -> Self::Response {
        match op {
            OpWr::Inc => {
                Ok(self.inc())
            }
        }
    }
}

impl DsInterface<'a> for NodeReplicated<NrCounter> {
    type ThreadToken = ThreadToken;
    type RdOp<'a> = <NrCounter as Dispatch>::ReadOperation<'a>;
    type WrOp = <NrCounter as Dispatch>::WriteOperation;
    type Resp = <NrCounter as Dispatch>::Response;
    type Disp = NrCounter;

    /// Allocate a new data-structure.
    ///
    /// - `replicas`: How many replicas the data-structure should maintain.
    /// - `logs`: How many logs the data-structure should be partitioned over.
    fn new(replicas: NonZeroUsize, logs: NonZeroUsize, log_size: usize) -> Arc<Self>
    {

    }

    /// Register a thread with a data-structure.
    ///
    /// - `rid` indicates which replica the thread should use.
    fn register(&self, rid: ReplicaId) -> Option<Self::ThreadToken>
    {

    }

    /// Apply a mutable operation to the data-structure.
    fn execute_mut(
        &self,
        op: Self::WrOp,
        idx: Self::ThreadToken,
    ) -> Self::Resp
    {

    }

    /// Apply a immutable operation to the data-structure.
    fn execute(
        &self,
        op: Self::RdOp,
        idx: Self::ThreadToken,
    ) -> Self::Resp
    {

    }
}


/// Generate a random sequence of operations
///
/// # Arguments
///  - `nop`: Number of operations to generate
///  - `write`: true will Put, false will generate Get sequences
///  - `span`: Maximum key
///  - `distribution`: Supported distribution 'uniform' or 'skewed'
pub fn generate_operations<R: DsInterface>(
    nop: usize,
    write_ratio: usize,
) -> Vec<Operation<<R as DsInterface>::RdOp, <R as DsInterface>::WrOp>> {
    let mut ops = Vec::with_capacity(nop);

    let mut t_rng = rand::thread_rng();

    for idx in 0..nop {
        if idx % 100 < write_ratio {
            ops.push(Operation::WriteOperation(OpWr::Inc));
        } else {
            ops.push(Operation::ReadOperation(OpRd::Get));
        }
    }

    ops.shuffle(&mut t_rng);
    ops
}

/// Compare scale-out behaviour of synthetic data-structure.
fn counter_scale_out<R>(c: &mut TestHarness, name: &str, write_ratio: usize)
where
    R: DsInterface + Send + Sync + 'static
{
    let ops = generate_operations(NOP, write_ratio);
    let bench_name = format!("{}-scaleout-wr{}", name, write_ratio);

    mkbench::ScaleBenchBuilder::<R>::new(ops)
        .thread_defaults()
        //.threads(1)
        //.threads(73)
        //.threads(96)
        //.threads(192)
        .update_batch(32)
        .log_size(2 * 1024 * 1024)
        .replica_strategy(mkbench::ReplicaStrategy::One)
        .replica_strategy(mkbench::ReplicaStrategy::Socket)
        .thread_mapping(ThreadMapping::Interleave)
        .log_strategy(mkbench::LogStrategy::One)
        .configure(
            c,
            &bench_name,
            |_cid, tkn, replica, op, _batch_size| match op {
                Operation::ReadOperation(op) => {
                    replica.execute(*op, tkn);
                }
                Operation::WriteOperation(op) => {
                    replica.execute_mut(*op, tkn);
                }
            },
        );
}

fn main() {
    let _r = env_logger::try_init();
    if cfg!(feature = "smokebench") {
        warn!("Running with feature 'smokebench' may not get the desired results");
    }

    bench_utils::disable_dvfs();

    let mut harness = Default::default();

    let write_ratios = if cfg!(feature = "exhaustive") {
        vec![0, 10, 20, 40, 60, 80, 100]
    } else if cfg!(feature = "smokebench") {
        vec![10]
    } else {
        vec![0, 10, 100]
    };

    //hashmap_single_threaded(&mut harness);
    for write_ratio in write_ratios.into_iter() {
        counter_scale_out::<NrCounter>(&mut harness, "counter", write_ratio);
    }
}