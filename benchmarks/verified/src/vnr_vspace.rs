// HashMap Benchmark for upstream NR
// Adapted from https://github.com/vmware/node-replication/blob/master/node-replication/benches/hashmap/main.rs
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Defines a hash-map that can be replicated.
#![allow(dead_code)]
// #![feature(generic_associated_types)]

use std::fmt::Debug;
use std::marker::Sync;
use std::num::NonZeroUsize;
use std::time::Duration;

use logging::warn;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use bench_utils::benchmark::*;
use bench_utils::mkbench::{self, DsInterface};
use bench_utils::topology::ThreadMapping;
use bench_utils::Operation;
use verified_node_replication::{
    AffinityFn, Dispatch, NodeReplicated, NodeReplicatedT, ReplicaId, ThreadToken,
};

use builtin::Tracked;

// Number of operation for test-harness.
#[cfg(feature = "smokebench")]
pub const NOP: usize = 2_500_000;
#[cfg(not(feature = "smokebench"))]
pub const NOP: usize = 25_000_000;

// ! Evaluates a virtual address space implementation using node-replication.
//#![feature(test)]
//#![feature(bench_black_box)]
// #![crate_type = "staticlib"]
extern crate alloc;

use std::fmt;
use std::mem::transmute;
use std::pin::Pin;

use logging::{debug, trace};
use x86::bits64::paging::*;

const VSPACE_RANGE: u64 = 512 * 1024 * 1024 * 1024;

fn kernel_vaddr_to_paddr(v: VAddr) -> PAddr {
    let vaddr_val: usize = v.into();
    PAddr::from(vaddr_val as u64 - 0x0)
}

fn paddr_to_kernel_vaddr(p: PAddr) -> VAddr {
    let paddr_val: u64 = p.into();
    VAddr::from((paddr_val + 0x0) as usize)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VSpaceError {
    pub at: u64,
}

/// Type of resource we're trying to allocate
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ResourceType {
    /// ELF Binary data
    Binary,
    /// Physical memory
    Memory,
    /// Page-table meta-data
    PageTable,
}

/// Mapping rights to give to address translation.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[allow(unused)]
pub enum MapAction {
    /// Don't map
    None,
    /// Map region read-only.
    ReadUser,
    /// Map region read-only for kernel.
    ReadKernel,
    /// Map region read-write.
    ReadWriteUser,
    /// Map region read-write for kernel.
    ReadWriteKernel,
    /// Map region read-executable.
    ReadExecuteUser,
    /// Map region read-executable for kernel.
    ReadExecuteKernel,
    /// Map region read-write-executable.
    ReadWriteExecuteUser,
    /// Map region read-write-executable for kernel.
    ReadWriteExecuteKernel,
}

impl MapAction {
    /// Transform MapAction into rights for 1 GiB page.
    fn to_pdpt_rights(&self) -> PDPTFlags {
        use MapAction::*;
        match self {
            None => PDPTFlags::empty(),
            ReadUser => PDPTFlags::XD,
            ReadKernel => PDPTFlags::US | PDPTFlags::XD,
            ReadWriteUser => PDPTFlags::RW | PDPTFlags::XD,
            ReadWriteKernel => PDPTFlags::RW | PDPTFlags::US | PDPTFlags::XD,
            ReadExecuteUser => PDPTFlags::empty(),
            ReadExecuteKernel => PDPTFlags::US,
            ReadWriteExecuteUser => PDPTFlags::RW,
            ReadWriteExecuteKernel => PDPTFlags::RW | PDPTFlags::US,
        }
    }

    /// Transform MapAction into rights for 2 MiB page.
    fn to_pd_rights(&self) -> PDFlags {
        use MapAction::*;
        match self {
            None => PDFlags::empty(),
            ReadUser => PDFlags::XD,
            ReadKernel => PDFlags::US | PDFlags::XD,
            ReadWriteUser => PDFlags::RW | PDFlags::XD,
            ReadWriteKernel => PDFlags::RW | PDFlags::US | PDFlags::XD,
            ReadExecuteUser => PDFlags::empty(),
            ReadExecuteKernel => PDFlags::US,
            ReadWriteExecuteUser => PDFlags::RW,
            ReadWriteExecuteKernel => PDFlags::RW | PDFlags::US,
        }
    }

    /// Transform MapAction into rights for 4KiB page.
    fn to_pt_rights(&self) -> PTFlags {
        use MapAction::*;
        match self {
            None => PTFlags::empty(),
            ReadUser => PTFlags::XD,
            ReadKernel => PTFlags::US | PTFlags::XD,
            ReadWriteUser => PTFlags::RW | PTFlags::XD,
            ReadWriteKernel => PTFlags::RW | PTFlags::US | PTFlags::XD,
            ReadExecuteUser => PTFlags::empty(),
            ReadExecuteKernel => PTFlags::US,
            ReadWriteExecuteUser => PTFlags::RW,
            ReadWriteExecuteKernel => PTFlags::RW | PTFlags::US,
        }
    }
}

impl fmt::Display for MapAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use MapAction::*;
        match self {
            None => write!(f, " ---"),
            ReadUser => write!(f, "uR--"),
            ReadKernel => write!(f, "kR--"),
            ReadWriteUser => write!(f, "uRW-"),
            ReadWriteKernel => write!(f, "kRW-"),
            ReadExecuteUser => write!(f, "uR-X"),
            ReadExecuteKernel => write!(f, "kR-X"),
            ReadWriteExecuteUser => write!(f, "uRWX"),
            ReadWriteExecuteKernel => write!(f, "kRWX"),
        }
    }
}

pub struct VSpace {
    pub pml4: Pin<Box<PML4>>,
    pub mem_counter: usize,
    mapping: mmap::MemoryMap,
    mem_ptr: *mut u8,
    //allocs: Vec<(*mut u8, usize)>,
}

unsafe impl Sync for VSpace {}
unsafe impl Send for VSpace {}

/// We support a mutable put operation on the hashmap.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Modify {
    Map(u64, u64),
}

/// We support an immutable read operation to lookup a key from the hashmap.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Access {
    Resolve(u64),
}

/// The Dispatch traits executes `ReadOperation` (our Access enum)
/// and `WriteOperation` (our Modify enum) against the replicated
/// data-structure.
impl Dispatch for VSpace {
    type ReadOperation = Access;
    type WriteOperation = Modify;
    type Response = u64;
    type View = VSpace;

    fn init() -> Self {
        Default::default()
    }

    /// The `dispatch` function applies the immutable operations.
    fn dispatch(&self, op: Self::ReadOperation) -> Self::Response {
        match op {
            Access::Resolve(key) => self.resolve_wrapped(key),
        }
    }

    /// The `dispatch_mut` function applies the mutable operations.
    fn dispatch_mut(&mut self, op: Self::WriteOperation) -> Self::Response {
        match op {
            Modify::Map(key, value) => self.map_generic_wrapped(key, value, 0x1000) as u64,
        }
    }

    // partial eq also add an exec operation
    fn clone_write_op(op: &Self::WriteOperation) -> Self::WriteOperation {
        op.clone()
    }

    fn clone_response(op: &Self::Response) -> Self::Response {
        op.clone()
    }
}

// impl<T, U> SomeTrait for T
//    where T: AnotherTrait<AssocType=U>
struct VNRWrapper {
    val: NodeReplicated<VSpace>,
}

/// The interface a data-structure must implement to be benchmarked by
/// `ScaleBench`.
impl DsInterface for VNRWrapper {
    type D = VSpace; //: Dispatch + Default + Sync;

    /// Allocate a new data-structure.
    ///
    /// - `replicas`: How many replicas the data-structure should maintain.
    /// - `logs`: How many logs the data-structure should be partitioned over.
    fn new(replicas: NonZeroUsize, _logs: NonZeroUsize, _log_size: usize) -> Self {
        VNRWrapper {
            val: NodeReplicatedT::<Self::D>::new(
                replicas.into(),
                AffinityFn::new(mkbench::chg_affinity),
            ),
        }
    }

    /// Register a thread with a data-structure.
    ///
    /// - `rid` indicates which replica the thread should use.
    fn register(&mut self, rid: ReplicaId) -> Option<ThreadToken<Self::D>> {
        NodeReplicatedT::<Self::D>::register(&mut self.val, rid)
    }

    /// Apply a mutable operation to the data-structure.
    fn execute_mut(
        &self,
        op: <Self::D as Dispatch>::WriteOperation,
        idx: ThreadToken<Self::D>,
    ) -> Result<(<Self::D as Dispatch>::Response, ThreadToken<Self::D>), ThreadToken<Self::D>> {
        match NodeReplicatedT::execute_mut(&self.val, op, idx, Tracked::assume_new()) {
            Ok((res, tkn, _)) => Ok((res, tkn)),
            Err((tkn, _)) => Err(tkn),
        }
    }

    /// Apply a immutable operation to the data-structure.
    fn execute(
        &self,
        op: <Self::D as Dispatch>::ReadOperation,
        idx: ThreadToken<Self::D>,
    ) -> Result<(<Self::D as Dispatch>::Response, ThreadToken<Self::D>), ThreadToken<Self::D>> {
        match NodeReplicatedT::execute(&self.val, op, idx, Tracked::assume_new()) {
            Ok((res, tkn, _)) => Ok((res, tkn)),
            Err((tkn, _)) => Err(tkn),
        }
    }
}

/*
       pub fn map_generic_wrapped(
           self: &mut VSpace,
           vbase: u64,
           pregion: u64,
           pregion_len: usize,
           //rights: &MapAction,
       ) -> bool;

       pub fn resolve_wrapped(self: &mut VSpace, vbase: u64) -> u64;
*/

impl Drop for VSpace {
    fn drop(&mut self) {
        /*unsafe {
            self.allocs.reverse();
            for (base, size) in self.allocs.iter() {
                //println!("-- dealloc {:p} {:#x}", base, size);
                alloc::alloc::dealloc(
                    *base,
                    core::alloc::Layout::from_size_align_unchecked(*size, 4096),
                );
            }
        }*/
    }
}

pub const TWO_MIB: usize = 2 * 1024 * 1024;
pub const ONE_GIB: usize = 1024 * 1024 * 1024;

// sudo sh -c "echo 16 > /sys/devices/system/node/node0/hugepages/hugepages-1048576kB/nr_hugepages"
// sudo sh -c "echo 16 > /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/nr_hugepages"
// sudo sh -c "echo 16 > /sys/devices/system/node/node2/hugepages/hugepages-1048576kB/nr_hugepages"
// sudo sh -c "echo 16 > /sys/devices/system/node/node3/hugepages/hugepages-1048576kB/nr_hugepages"

pub fn alloc(size: usize, ps: usize) -> mmap::MemoryMap {
    use libc::{MAP_ANON, MAP_HUGETLB, MAP_POPULATE, MAP_SHARED};

    const MAP_HUGE_SHIFT: usize = 26;
    const MAP_HUGE_2MB: i32 = 21 << MAP_HUGE_SHIFT;
    const MAP_HUGE_1GB: i32 = 30 << MAP_HUGE_SHIFT;

    pub const FOUR_KIB: usize = 4 * 1024;
    const PAGESIZE: u64 = FOUR_KIB as u64;

    assert!(size % FOUR_KIB == 0 || size % TWO_MIB == 0 || size % ONE_GIB == 0);

    let mut non_standard_flags = MAP_SHARED | MAP_ANON | MAP_POPULATE;
    match ps {
        TWO_MIB => non_standard_flags |= MAP_HUGETLB | MAP_HUGE_2MB,
        ONE_GIB => non_standard_flags |= MAP_HUGETLB | MAP_HUGE_1GB,
        _ => (),
    }

    let flags = [
        mmap::MapOption::MapNonStandardFlags(non_standard_flags),
        mmap::MapOption::MapReadable,
        mmap::MapOption::MapWritable,
    ];
    let res = mmap::MemoryMap::new(size, &flags).expect("can't allocate?");
    if res.data().is_null() {
        panic!("can't get memory, do we have reserved huge-pages?");
    }

    // Make sure memory is not swapped:
    //let lock_ret = unsafe { libc::mlock(res.data() as *const libc::c_void, res.len()) };
    //if lock_ret == -1 {
    //    panic!("can't mlock mem");
    //}
    //assert!(lock_ret == 0);

    res
}

impl Default for VSpace {
    fn default() -> VSpace {
        let mapping = alloc(3 * ONE_GIB, ONE_GIB);
        let mem_ptr = mapping.data();

        // make sure the memory for ptable is some contiguous block
        // this allows Linux / THP to kick in and increase tput by ~60Mops
        // make sure to do:
        // sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
        //let mem_ptr = unsafe { alloc::alloc::alloc(core::alloc::Layout::from_size_align_unchecked(1075851264, 4096)) };

        let mut vs = VSpace {
            pml4: Box::pin(
                [PML4Entry::new(PAddr::from(0x0u64), PML4Flags::empty()); PAGE_SIZE_ENTRIES],
            ),
            mapping,
            mem_counter: 4096,
            mem_ptr, //allocs: Vec::with_capacity(1024),
        };
        for i in 0..VSPACE_RANGE / 4096 {
            assert!(vs
                .map_generic(
                    VAddr::from(i * 4096),
                    (PAddr::from(i * 4096), 4096),
                    MapAction::ReadWriteExecuteUser,
                )
                .is_ok());
        }

        logging::error!("vs.mem_counter {}", vs.mem_counter);

        vs
    }
}

impl VSpace {
    pub fn map_generic_wrapped(
        self: &mut VSpace,
        vbase: u64,
        pregion: u64,
        pregion_len: usize,
    ) -> bool {
        let rights = MapAction::ReadWriteExecuteUser;
        let r = self.map_generic(
            VAddr::from(vbase),
            (PAddr::from(pregion), pregion_len),
            rights,
        );

        r.is_ok()
    }

    pub fn map_generic(
        &mut self,
        vbase: VAddr,
        pregion: (PAddr, usize),
        rights: MapAction,
    ) -> Result<(), VSpaceError> {
        let (pbase, psize) = pregion;
        assert_eq!(pbase % BASE_PAGE_SIZE, 0);
        assert_eq!(psize % BASE_PAGE_SIZE, 0);
        assert_eq!(vbase % BASE_PAGE_SIZE, 0);
        assert_ne!(rights, MapAction::None);

        debug!(
            "map_generic {:#x} -- {:#x} -> {:#x} -- {:#x} {}",
            vbase,
            vbase + psize,
            pbase,
            pbase + psize,
            rights
        );

        let pml4_idx = pml4_index(vbase);
        if !self.pml4[pml4_idx].is_present() {
            trace!("New PDPDT for {:?} @ PML4[{}]", vbase, pml4_idx);
            self.pml4[pml4_idx] = self.new_pdpt();
        }
        assert!(
            self.pml4[pml4_idx].is_present(),
            "The PML4 slot we need was not allocated?"
        );

        let pdpt = self.get_pdpt(self.pml4[pml4_idx]);
        let mut pdpt_idx = pdpt_index(vbase);
        // TODO: if we support None mappings, this is if not good enough:
        if !pdpt[pdpt_idx].is_present() {
            // The virtual address corresponding to our position within the page-table
            let vaddr_pos: usize = PML4_SLOT_SIZE * pml4_idx + HUGE_PAGE_SIZE * pdpt_idx;

            // In case we can map something at a 1 GiB granularity and
            // we still have at least 1 GiB to map, create huge-page mappings
            if vbase.as_usize() == vaddr_pos
                && (pbase % HUGE_PAGE_SIZE == 0)
                && psize >= HUGE_PAGE_SIZE
            {
                // To track how much space we've covered
                let mut mapped = 0;

                // Add entries to PDPT as long as we're within this allocated PDPT table
                // and have 1 GiB chunks to map:
                while mapped < psize && ((psize - mapped) >= HUGE_PAGE_SIZE) && pdpt_idx < 512 {
                    assert!(!pdpt[pdpt_idx].is_present());
                    pdpt[pdpt_idx] = PDPTEntry::new(
                        pbase + mapped,
                        PDPTFlags::P | PDPTFlags::PS | rights.to_pdpt_rights(),
                    );
                    trace!(
                        "Mapped 1GiB range {:#x} -- {:#x} -> {:#x} -- {:#x}",
                        vbase + mapped,
                        (vbase + mapped) + HUGE_PAGE_SIZE,
                        pbase + mapped,
                        (vbase + mapped) + HUGE_PAGE_SIZE
                    );

                    pdpt_idx += 1;
                    mapped += HUGE_PAGE_SIZE;
                }

                if mapped < psize {
                    trace!(
                        "map_generic recurse from 1 GiB map to finish {:#x} -- {:#x} -> {:#x} -- {:#x}",
                        vbase + mapped,
                        vbase + (psize - mapped),
                        (pbase + mapped),
                        pbase + (psize - mapped),
                    );
                    return self.map_generic(
                        vbase + mapped,
                        ((pbase + mapped), psize - mapped),
                        rights,
                    );
                } else {
                    // Everything fit in 1 GiB ranges,
                    // We're done with mappings
                    return Ok(());
                }
            } else {
                trace!(
                    "Mapping 0x{:x} -- 0x{:x} is smaller than 1 GiB, going deeper.",
                    vbase,
                    vbase + psize
                );
                pdpt[pdpt_idx] = self.new_pd();
            }
        }
        assert!(
            pdpt[pdpt_idx].is_present(),
            "The PDPT entry we're relying on is not allocated?"
        );
        if pdpt[pdpt_idx].is_page() {
            // "An existing mapping already covers the 1 GiB range we're trying to map in?
            return Err(VSpaceError { at: vbase.as_u64() });
        }

        let pd = self.get_pd(pdpt[pdpt_idx]);
        let mut pd_idx = pd_index(vbase);
        if !pd[pd_idx].is_present() {
            let vaddr_pos: usize =
                PML4_SLOT_SIZE * pml4_idx + HUGE_PAGE_SIZE * pdpt_idx + LARGE_PAGE_SIZE * pd_idx;

            // In case we can map something at a 2 MiB granularity and
            // we still have at least 2 MiB to map create large-page mappings
            if vbase.as_usize() == vaddr_pos
                && (pbase % LARGE_PAGE_SIZE == 0)
                && psize >= LARGE_PAGE_SIZE
            {
                let mut mapped = 0;
                // Add entries as long as we are within this allocated PDPT table
                // and have at least 2 MiB things to map
                while mapped < psize && ((psize - mapped) >= LARGE_PAGE_SIZE) && pd_idx < 512 {
                    if pd[pd_idx].is_present() {
                        trace!("Already mapped pd at {:#x}", pbase + mapped);
                        return Err(VSpaceError { at: vbase.as_u64() });
                    }

                    pd[pd_idx] = PDEntry::new(
                        pbase + mapped,
                        PDFlags::P | PDFlags::PS | rights.to_pd_rights(),
                    );
                    trace!(
                        "Mapped 2 MiB region {:#x} -- {:#x} -> {:#x} -- {:#x}",
                        vbase + mapped,
                        (vbase + mapped) + LARGE_PAGE_SIZE,
                        pbase + mapped,
                        (pbase + mapped) + LARGE_PAGE_SIZE
                    );

                    pd_idx += 1;
                    mapped += LARGE_PAGE_SIZE;
                }

                if mapped < psize {
                    trace!(
                        "map_generic recurse from 2 MiB map to finish {:#x} -- {:#x} -> {:#x} -- {:#x}",
                        vbase + mapped,
                        vbase + (psize - mapped),
                        (pbase + mapped),
                        pbase + (psize - mapped),
                    );
                    return self.map_generic(
                        vbase + mapped,
                        ((pbase + mapped), psize - mapped),
                        rights,
                    );
                } else {
                    // Everything fit in 2 MiB ranges,
                    // We're done with mappings
                    return Ok(());
                }
            } else {
                trace!(
                    "Mapping 0x{:x} -- 0x{:x} is smaller than 2 MiB, going deeper.",
                    vbase,
                    vbase + psize
                );
                pd[pd_idx] = self.new_pt();
            }
        }
        assert!(
            pd[pd_idx].is_present(),
            "The PD entry we're relying on is not allocated?"
        );
        if pd[pd_idx].is_page() {
            // An existing mapping already covers the 2 MiB range we're trying to map in?
            return Err(VSpaceError { at: vbase.as_u64() });
        }

        let pt = self.get_pt(pd[pd_idx]);
        let mut pt_idx = pt_index(vbase);
        let mut mapped: usize = 0;
        while mapped < psize && pt_idx < 512 {
            // XXX: allow updates
            //if !pt[pt_idx].is_present() {
            pt[pt_idx] = PTEntry::new(pbase + mapped, PTFlags::P | rights.to_pt_rights());
            //} else {
            //    return Err(VSpaceError { at: vbase.as_u64() });
            //}

            mapped += BASE_PAGE_SIZE;
            pt_idx += 1;
        }

        // Need go to different PD/PDPT/PML4 slot
        if mapped < psize {
            trace!(
                "map_generic recurse from 4 KiB map to finish {:#x} -- {:#x} -> {:#x} -- {:#x}",
                vbase + mapped,
                vbase + (psize - mapped),
                (pbase + mapped),
                pbase + (psize - mapped),
            );
            return self.map_generic(vbase + mapped, ((pbase + mapped), psize - mapped), rights);
        } else {
            // else we're done here, return
            Ok(())
        }
    }

    /// A simple wrapper function for allocating just one page.
    fn allocate_one_page(&mut self) -> PAddr {
        logging::info!("allocate a page...");
        self.mem_counter += 4096;
        self.allocate_pages(1, ResourceType::PageTable)
    }

    fn allocate_pages(&mut self, how_many: usize, _typ: ResourceType) -> PAddr {
        logging::info!("allocate_pages {}...", how_many);

        let new_region: *mut u8 = unsafe {
            /*alloc::alloc::alloc(core::alloc::Layout::from_size_align_unchecked(
                how_many * BASE_PAGE_SIZE,
                4096,
            ))*/
            assert!(self.mem_counter < 3 * ONE_GIB); // if this triggers you need to adjust the alloc size of `mem_ptr`
            self.mem_ptr.offset(self.mem_counter as isize)
        };
        self.mem_counter += how_many * 4096;

        assert!(!new_region.is_null());
        for i in 0..how_many * BASE_PAGE_SIZE {
            unsafe {
                *new_region.offset(i as isize) = 0u8;
            }
        }
        //self.allocs.push((new_region, how_many * BASE_PAGE_SIZE));

        kernel_vaddr_to_paddr(VAddr::from(new_region as usize))
    }

    fn new_pt(&mut self) -> PDEntry {
        let paddr: PAddr = self.allocate_one_page();
        return PDEntry::new(paddr, PDFlags::P | PDFlags::RW | PDFlags::US);
    }

    fn new_pd(&mut self) -> PDPTEntry {
        let paddr: PAddr = self.allocate_one_page();
        return PDPTEntry::new(paddr, PDPTFlags::P | PDPTFlags::RW | PDPTFlags::US);
    }

    fn new_pdpt(&mut self) -> PML4Entry {
        let paddr: PAddr = self.allocate_one_page();
        return PML4Entry::new(paddr, PML4Flags::P | PML4Flags::RW | PML4Flags::US);
    }

    /// Resolve a PDEntry to a page table.
    fn get_pt<'b>(&self, entry: PDEntry) -> &'b mut PT {
        unsafe { transmute::<VAddr, &mut PT>(paddr_to_kernel_vaddr(entry.address())) }
    }

    /// Resolve a PDPTEntry to a page directory.
    fn get_pd<'b>(&self, entry: PDPTEntry) -> &'b mut PD {
        unsafe { transmute::<VAddr, &mut PD>(paddr_to_kernel_vaddr(entry.address())) }
    }

    /// Resolve a PML4Entry to a PDPT.
    fn get_pdpt<'b>(&self, entry: PML4Entry) -> &'b mut PDPT {
        unsafe { transmute::<VAddr, &mut PDPT>(paddr_to_kernel_vaddr(entry.address())) }
    }

    pub fn resolve_wrapped(&self, addr: u64) -> u64 {
        let a = self
            .resolve_addr(VAddr::from(addr))
            .map(|pa| pa.as_u64())
            .unwrap_or(0x0);
        //log::error!("{:#x} -> {:#x}", addr, a);
        a
    }

    pub fn resolve_addr(&self, addr: VAddr) -> Option<PAddr> {
        //log::error!("resolv addr {:#x}", addr);
        let pml4_idx = pml4_index(addr);
        if self.pml4[pml4_idx].is_present() {
            let pdpt_idx = pdpt_index(addr);
            let pdpt = self.get_pdpt(self.pml4[pml4_idx]);
            if pdpt[pdpt_idx].is_present() {
                if pdpt[pdpt_idx].is_page() {
                    // Page is a 1 GiB mapping, we have to return here
                    // let page_offset = addr.huge_page_offset();
                    unreachable!("dont go here");
                    // return Some(pdpt[pdpt_idx].address() + page_offset);
                } else {
                    let pd_idx = pd_index(addr);
                    let pd = self.get_pd(pdpt[pdpt_idx]);
                    if pd[pd_idx].is_present() {
                        if pd[pd_idx].is_page() {
                            // Encountered a 2 MiB mapping, we have to return here
                            // let page_offset = addr.large_page_offset();
                            unreachable!("dont go here");
                            // return Some(pd[pd_idx].address() + page_offset);
                        } else {
                            let pt_idx = pt_index(addr);
                            let pt = self.get_pt(pd[pd_idx]);
                            if pt[pt_idx].is_present() {
                                //log::error!("return {:#x}", pt[pt_idx].address());

                                let page_offset = addr.base_page_offset();
                                return Some(pt[pt_idx].address() + page_offset);
                            }
                        }
                    }
                }
            }
        } else {
            // log::error!("pml4 not present {:#x}", addr);
            unreachable!("dont go here");
        }

        unreachable!("dont go here");
    }

    pub fn map_new(
        &mut self,
        base: VAddr,
        size: usize,
        rights: MapAction,
        paddr: PAddr,
    ) -> Result<(PAddr, usize), VSpaceError> {
        assert_eq!(base % BASE_PAGE_SIZE, 0, "base is not page-aligned");
        assert_eq!(size % BASE_PAGE_SIZE, 0, "size is not page-aligned");
        self.map_generic(base, (paddr, size), rights)?;
        Ok((paddr, size))
    }
}

/// Generate a random sequence of operations
///
/// # Arguments
///  - `nop`: Number of operations to generate
///  - `write`: true will Put, false will generate Get sequences
///  - `span`: Maximum key
///  - `distribution`: Supported distribution 'uniform' or 'skewed'
pub fn generate_operations(nop: usize, write_ratio: usize) -> Vec<Operation<Access, Modify>> {
    let mut ops = Vec::with_capacity(nop);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    const MASK: u64 = 0x7fffffffff & !0xfffu64;
    const PAGE_RANGE_MASK: u64 = !0xffff_0000_0000_0fff;
    const MAP_SIZE_MASK: u64 = !0xffff_ffff_f000_0fff;
    for idx in 0..nop {
        if idx % 100 < write_ratio {
            ops.push(Operation::WriteOperation(Modify::Map(
                rng.gen::<u64>() & MASK,
                rng.gen::<u64>() & MASK,
            )))
        } else {
            ops.push(Operation::ReadOperation(Access::Resolve(
                rng.gen::<u64>() & MASK,
            )))
        }
    }

    ops.shuffle(&mut rng);
    ops
}

fn main() {
    let _r = env_logger::try_init();
    if cfg!(feature = "smokebench") {
        warn!("Running with feature 'smokebench' may not get the desired results");
    }

    bench_utils::disable_dvfs();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        println!("Usage: cargo run -- n_threads reads_pct, runtime, numa_policy, run_id_num");
    }

    let n_threads = args[1].parse::<usize>().unwrap();
    let reads_pct = args[2].parse::<usize>().unwrap();
    let write_ratio = 100 - reads_pct;
    let runtime = args[3].parse::<u64>().unwrap();
    let numa_policy = match args[4].as_str() {
        "fill" => ThreadMapping::NUMAFill,
        "interleave" => ThreadMapping::Interleave,
        _ => panic!("supply fill or interleave as numa mapping"),
    };
    let run_id_num = &args[5];

    let mut harness = TestHarness::new(Duration::from_secs(runtime));

    let ops = generate_operations(NOP, write_ratio);
    let bench_name = format!(
        "vnr_vspace-{}-{}-{}-{}",
        n_threads, write_ratio, numa_policy, run_id_num
    );

    mkbench::ScaleBenchBuilder::<VNRWrapper>::new(ops)
        .threads(n_threads)
        .update_batch(32)
        .log_size(2 * 1024 * 1024)
        .replica_strategy(mkbench::ReplicaStrategy::Socket)
        .thread_mapping(numa_policy)
        .read_pct(reads_pct)
        .log_strategy(mkbench::LogStrategy::One)
        .configure(
            &mut harness,
            &bench_name,
            |_cid, tkn, replica, op, _batch_size| match op {
                Operation::ReadOperation(op) => match replica.execute(*op, tkn) {
                    Ok(r) => r.1,
                    Err(r) => r,
                },
                Operation::WriteOperation(op) => match replica.execute_mut(*op, tkn) {
                    Ok(r) => r.1,
                    Err(r) => r,
                },
            },
        );
}
