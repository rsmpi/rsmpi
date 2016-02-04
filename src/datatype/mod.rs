//! Describing data
//!
//! The core function of MPI is getting data from point A to point B (where A and B are e.g. single
//! processes, multiple processes, the filesystem, ...). It offers facilities to describe that data
//! (layout in memory, behavior under certain operators) that go beyound a start address and a
//! number of bytes.
//!
//! An MPI datatype describes a memory layout and semantics (e.g. in a collective reduce
//! operation). There are several pre-defined `SystemDatatype`s which directly correspond to Rust
//! primitive types, such as `MPI_DOUBLE` and `f64`. A direct relationship between a Rust type and
//! an MPI datatype is covered by the `EquivalentDatatype` trait. Starting from the
//! `SystemDatatype`s, the user can build various `UserDatatype`s, e.g. to describe the layout of a
//! struct (which should then implement `EquivalentDatatype`) or to intrusively describe parts of
//! an object in memory like all elements below the diagonal of a dense matrix stored in row-major
//! order.
//!
//! A `Buffer` describes a specific piece of data in memory that MPI should operate on. In addition
//! to specifying the datatype of the data. It knows the address in memory where the data begins
//! and how many instances of the datatype are contained in the data. The `Buffer` trait is
//! implemented for slices that contain types implementing `EquivalentDatatype`.
//!
//! In order to use arbitrary datatypes to describe the contents of a slice, the `View` type is
//! provided. However, since it can be used to instruct the underlying MPI implementation to
//! rummage around arbitrary parts of memory, its constructors are currently marked unsafe.
//!
//! # Unfinished features
//!
//! - **4.1.2**: Datatype constructors, `MPI_Type_create_struct()`
//! - **4.1.3**: Subarray datatype constructors, `MPI_Type_create_subarray()`,
//! - **4.1.4**: Distributed array datatype constructors, `MPI_Type_create_darray()`
//! - **4.1.5**: Address and size functions, `MPI_Get_address()`, `MPI_Aint_add()`,
//! `MPI_Aint_diff()`, `MPI_Type_size()`, `MPI_Type_size_x()`
//! - **4.1.7**: Extent and bounds of datatypes: `MPI_Type_get_extent()`,
//! `MPI_Type_get_extent_x()`, `MPI_Type_create_resized()`
//! - **4.1.8**: True extent of datatypes, `MPI_Type_get_true_extent()`,
//! `MPI_Type_get_true_extent_x()`
//! - **4.1.10**: Duplicating a datatype, `MPI_Type_dup()`
//! - **4.1.11**: `MPI_Get_elements()`, `MPI_Get_elements_x()`
//! - **4.1.13**: Decoding a datatype, `MPI_Type_get_envelope()`, `MPI_Type_get_contents()`
//! - **4.2**: Pack and unpack, `MPI_Pack()`, `MPI_Unpack()`, `MPI_Pack_size()`
//! - **4.3**: Canonical pack and unpack, `MPI_Pack_external()`, `MPI_Unpack_external()`,
//! `MPI_Pack_external_size()`

use std::{mem};
use std::os::raw::{c_void};

use conv::ConvUtil;

use super::{Address, Count};

use ffi;
use ffi::MPI_Datatype;

use raw::traits::*;

pub mod traits;

/// A system datatype, e.g. `MPI_FLOAT`
///
/// # Standard section(s)
///
/// 3.2.2
#[derive(Copy, Clone)]
pub struct SystemDatatype(MPI_Datatype);

impl AsRaw for SystemDatatype {
    type Raw = MPI_Datatype;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl RawDatatype for SystemDatatype { }

/// A direct equivalence exists between the implementing type and an MPI datatype
///
/// # Standard section(s)
///
/// 3.2.2
pub trait EquivalentDatatype {
    /// The type of the equivalent MPI datatype (e.g. `SystemDatatype` or `UserDatatype`)
    type Out: RawDatatype;
    /// The MPI datatype that is equivalent to this Rust type
    fn equivalent_datatype() -> Self::Out;
}

macro_rules! equivalent_system_datatype {
    ($rstype:path, $mpitype:path) => (
        impl EquivalentDatatype for $rstype {
            type Out = SystemDatatype;
            fn equivalent_datatype() -> Self::Out { SystemDatatype($mpitype) }
        }
    )
}

equivalent_system_datatype!(f32, ffi::RSMPI_FLOAT);
equivalent_system_datatype!(f64, ffi::RSMPI_DOUBLE);

equivalent_system_datatype!(i8, ffi::RSMPI_INT8_T);
equivalent_system_datatype!(i16, ffi::RSMPI_INT16_T);
equivalent_system_datatype!(i32, ffi::RSMPI_INT32_T);
equivalent_system_datatype!(i64, ffi::RSMPI_INT64_T);

equivalent_system_datatype!(u8, ffi::RSMPI_UINT8_T);
equivalent_system_datatype!(u16, ffi::RSMPI_UINT16_T);
equivalent_system_datatype!(u32, ffi::RSMPI_UINT32_T);
equivalent_system_datatype!(u64, ffi::RSMPI_UINT64_T);

/// A user defined MPI datatype
///
/// # Standard section(s)
///
/// 4
pub struct UserDatatype(MPI_Datatype);

impl UserDatatype {
    /// Constructs a new datatype by concatenating `count` repetitions of `oldtype`
    ///
    /// # Examples
    /// See `examples/contiguous.rs`
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn contiguous<D: RawDatatype>(count: Count, oldtype: D) -> UserDatatype {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_contiguous(count, oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Construct a new datatype out of `count` blocks of `blocklength` elements of `oldtype`
    /// concatenated with the start of consecutive blocks placed `stride` elements apart.
    ///
    /// # Examples
    /// See `examples/vector.rs`
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn vector<D: RawDatatype>(count: Count, blocklength: Count, stride: Count, oldtype: D) -> UserDatatype {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_vector(count, blocklength, stride, oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Like `vector()` but `stride` is given in bytes rather than elements of `oldtype`.
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn heterogeneous_vector<D: RawDatatype>(count: Count, blocklength: Count, stride: Address, oldtype: D) -> UserDatatype {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_hvector(count, blocklength, stride, oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Constructs a new type out of multiple blocks of individual length and displacement.
    /// Block `i` will be `blocklengths[i]` items of datytpe `oldtype` long and displaced by
    /// `dispplacements[i]` items of the `oldtype`.
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn indexed<D: RawDatatype>(blocklengths: &[Count], displacements: &[Count], oldtype: D) -> UserDatatype {
        assert_eq!(blocklengths.len(), displacements.len());
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_indexed(blocklengths.count(), blocklengths.as_ptr(),
                displacements.as_ptr(), oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Constructs a new type out of multiple blocks of individual length and displacement.
    /// Block `i` will be `blocklengths[i]` items of datytpe `oldtype` long and displaced by
    /// `dispplacements[i]` bytes.
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn heterogeneous_indexed<D: RawDatatype>(blocklengths: &[Count], displacements: &[Address], oldtype: D) -> UserDatatype {
        assert_eq!(blocklengths.len(), displacements.len());
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_create_hindexed(blocklengths.count(), blocklengths.as_ptr(),
                displacements.as_ptr(), oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Construct a new type out of blocks of the same length and individual displacements.
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn indexed_block<D: RawDatatype>(blocklength: Count, displacements: &[Count], oldtype: D) -> UserDatatype {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_create_indexed_block(displacements.count(), blocklength,
                displacements.as_ptr(), oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Construct a new type out of blocks of the same length and individual displacements.
    /// Displacements are in bytes.
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn heterogeneous_indexed_block<D: RawDatatype>(blocklength: Count, displacements: &[Address], oldtype: D) -> UserDatatype {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_create_hindexed_block(displacements.count(), blocklength,
                displacements.as_ptr(), oldtype.as_raw(), &mut newtype);
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }
}

impl Drop for UserDatatype {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Type_free(&mut self.0);
        }
        assert_eq!(self.0, ffi::RSMPI_DATATYPE_NULL);
    }
}

impl AsRaw for UserDatatype {
    type Raw = MPI_Datatype;
    unsafe fn as_raw(&self) -> Self::Raw { self.0 }
}

impl RawDatatype for UserDatatype { }

/// Something that has an associated datatype
// TODO: merge this into Buffer, maybe?
pub trait Datatype {
    /// The type of the associated MPI datatype (e.g. `SystemDatatype` or `UserDatatype`)
    type Out: RawDatatype;
    /// The associated MPI datatype
    fn datatype(&self) -> Self::Out;
}

impl<T> Datatype for T where T: EquivalentDatatype {
    type Out = <T as EquivalentDatatype>::Out;
    fn datatype(&self) -> Self::Out { <T as EquivalentDatatype>::equivalent_datatype() }
}

impl<T> Datatype for [T] where T: EquivalentDatatype {
    type Out = <T as EquivalentDatatype>::Out;
    fn datatype(&self) -> Self::Out { <T as EquivalentDatatype>::equivalent_datatype() }
}

/// A countable collection of things.
pub trait Collection {
    /// How many things are in this connection.
    fn count(&self) -> Count;
}

impl<T> Collection for T where T: EquivalentDatatype {
    fn count(&self) -> Count { 1 }
}

impl<T> Collection for [T] where T: EquivalentDatatype {
    fn count(&self) -> Count {
        self.len().value_as().expect("Length of slice cannot be expressed as an MPI Count.")
    }
}

/// Provides a pointer to the starting address in memory.
pub trait Pointer {
    /// A pointer to the starting address in memory
    unsafe fn pointer(&self) -> *const c_void;
}

impl<T> Pointer for T where T: EquivalentDatatype {
    unsafe fn pointer(&self) -> *const c_void { mem::transmute(self) }
}

impl<T> Pointer for [T] where T: EquivalentDatatype {
    unsafe fn pointer(&self) -> *const c_void { mem::transmute(self.as_ptr()) }
}

/// Provides a mutable pointer to the starting address in memory.
pub trait PointerMut {
    /// A mutable pointer to the starting address in memory
    unsafe fn pointer_mut(&mut self) -> *mut c_void;
}

impl<T> PointerMut for T where T: EquivalentDatatype {
    unsafe fn pointer_mut(&mut self) -> *mut c_void { mem::transmute(self) }
}

impl<T> PointerMut for [T] where T: EquivalentDatatype {
    unsafe fn pointer_mut(&mut self) -> *mut c_void { mem::transmute(self.as_mut_ptr()) }
}

/// A buffer is a region in memory that starts at `pointer()` and contains `count()` copies of
/// `datatype()`.
pub trait Buffer: Pointer + Collection + Datatype { }
impl<T> Buffer for T where T: EquivalentDatatype { }
impl<T> Buffer for [T] where T: EquivalentDatatype { }

/// A mutable buffer is a region in memory that starts at `pointer_mut()` and contains `count()`
/// copies of `datatype()`.
pub trait BufferMut: PointerMut + Collection + Datatype { }
impl<T> BufferMut for T where T: EquivalentDatatype { }
impl<T> BufferMut for [T] where T: EquivalentDatatype { }

/// A buffer with a user specified count and datatype
///
/// # Safety
///
/// Views can be used to instruct the underlying MPI library to rummage around at arbitrary
/// locations in memory. This might be controlled later on using datatype bounds an slice lengths
/// but for now, all View constructors are marked `unsafe`.
pub struct View<'d, 'b, D: 'd, B: 'b + ?Sized>
where D: RawDatatype, B: Pointer {
    datatype: &'d D,
    count: Count,
    buffer: &'b B
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> View<'d, 'b, D, B>
where D: RawDatatype, B: Pointer {
    /// Return a view of `buffer` containing `count` instances of MPI datatype `datatype`.
    ///
    /// # Examples
    /// See `examples/contiguous.rs`, `examples/vector.rs`
    pub unsafe fn with_count_and_datatype(buffer: &'b B, count: Count, datatype: &'d D) -> View<'d, 'b, D, B> {
        View { datatype: datatype, count: count, buffer: buffer }
    }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> Datatype for View<'d, 'b, D, B>
where D: RawDatatype, B: Pointer {
    type Out = &'d D;
    fn datatype(&self) -> Self::Out { self.datatype }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> Collection for View<'d, 'b, D, B>
where D: RawDatatype, B: Pointer {
    fn count(&self) -> Count { self.count }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> Pointer for View<'d, 'b, D, B>
where D: RawDatatype, B: Pointer {
    unsafe fn pointer(&self) -> *const c_void { self.buffer.pointer() }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> Buffer for View<'d, 'b, D, B>
where D: RawDatatype, B: Pointer { }

/// A buffer with a user specified count and datatype
///
/// # Safety
///
/// Views can be used to instruct the underlying MPI library to rummage around at arbitrary
/// locations in memory. This might be controlled later on using datatype bounds an slice lengths
/// but for now, all View constructors are marked `unsafe`.
pub struct MutView<'d, 'b, D: 'd, B: 'b + ?Sized>
where D: RawDatatype, B: PointerMut {
    datatype: &'d D,
    count: Count,
    buffer: &'b mut B
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> MutView<'d, 'b, D, B>
where D: RawDatatype, B: PointerMut {
    /// Return a view of `buffer` containing `count` instances of MPI datatype `datatype`.
    ///
    /// # Examples
    /// See `examples/contiguous.rs`, `examples/vector.rs`
    pub unsafe fn with_count_and_datatype(buffer: &'b mut B, count: Count, datatype: &'d D) -> MutView<'d, 'b, D, B> {
        MutView { datatype: datatype, count: count, buffer: buffer }
    }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> Datatype for MutView<'d, 'b, D, B>
where D: RawDatatype, B: PointerMut {
    type Out = &'d D;
    fn datatype(&self) -> Self::Out { self.datatype }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> Collection for MutView<'d, 'b, D, B>
where D: RawDatatype, B: PointerMut {
    fn count(&self) -> Count { self.count }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> PointerMut for MutView<'d, 'b, D, B>
where D: RawDatatype, B: PointerMut {
    unsafe fn pointer_mut(&mut self) -> *mut c_void { self.buffer.pointer_mut() }
}

impl<'d, 'b, D: 'd, B: 'b + ?Sized> BufferMut for MutView<'d, 'b, D, B>
where D: RawDatatype, B: PointerMut { }
