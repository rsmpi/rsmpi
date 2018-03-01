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
//! an MPI datatype is covered by the `Equivalence` trait. Starting from the
//! `SystemDatatype`s, the user can build various `UserDatatype`s, e.g. to describe the layout of a
//! struct (which should then implement `Equivalence`) or to intrusively describe parts of
//! an object in memory like all elements below the diagonal of a dense matrix stored in row-major
//! order.
//!
//! A `Buffer` describes a specific piece of data in memory that MPI should operate on. In addition
//! to specifying the datatype of the data. It knows the address in memory where the data begins
//! and how many instances of the datatype are contained in the data. The `Buffer` trait is
//! implemented for slices that contain types implementing `Equivalence`.
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

use std::{mem, slice};
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::os::raw::c_void;

use conv::ConvUtil;

use super::{Address, Count};

use ffi;
use ffi::MPI_Datatype;

use raw::traits::*;

/// Datatype traits
pub mod traits {
    pub use super::{AsDatatype, Buffer, BufferMut, Collection, Datatype, Equivalence, Partitioned,
                    PartitionedBuffer, PartitionedBufferMut, Pointer, PointerMut};
}

/// A reference to an MPI data type.
///
/// This is similar to a raw `MPI_Datatype` but is guaranteed to be a valid for `'a`.
#[derive(Copy, Clone, Debug)]
pub struct DatatypeRef<'a> {
    datatype: MPI_Datatype,
    phantom: PhantomData<&'a ()>,
}

unsafe impl<'a> AsRaw for DatatypeRef<'a> {
    type Raw = MPI_Datatype;
    fn as_raw(&self) -> Self::Raw {
        self.datatype
    }
}

impl<'a> Datatype for DatatypeRef<'a> {}

impl<'a> DatatypeRef<'a> {
    /// Wrap a raw handle.  The handle must remain valid for `'a`.
    pub unsafe fn from_raw(datatype: MPI_Datatype) -> Self {
        Self {
            datatype,
            phantom: PhantomData,
        }
    }
}

/// A system datatype, e.g. `MPI_FLOAT`
///
/// # Standard section(s)
///
/// 3.2.2
pub type SystemDatatype = DatatypeRef<'static>;

/// A direct equivalence exists between the implementing type and an MPI datatype
///
/// # Standard section(s)
///
/// 3.2.2
pub unsafe trait Equivalence {
    /// The type of the equivalent MPI datatype (e.g. `SystemDatatype` or `UserDatatype`)
    type Out: Datatype;
    /// The MPI datatype that is equivalent to this Rust type
    fn equivalent_datatype() -> Self::Out;
}

macro_rules! equivalent_system_datatype {
    ($rstype:path, $mpitype:path) => (
        unsafe impl Equivalence for $rstype {
            type Out = SystemDatatype;
            fn equivalent_datatype() -> Self::Out {
                unsafe { DatatypeRef::from_raw($mpitype) }
            }
        }
    )
}

equivalent_system_datatype!(bool, ffi::RSMPI_C_BOOL);

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

#[cfg(target_pointer_width = "32")]
equivalent_system_datatype!(usize, ffi::RSMPI_UINT32_T);
#[cfg(target_pointer_width = "32")]
equivalent_system_datatype!(isize, ffi::RSMPI_INT32_T);

#[cfg(target_pointer_width = "64")]
equivalent_system_datatype!(usize, ffi::RSMPI_UINT64_T);
#[cfg(target_pointer_width = "64")]
equivalent_system_datatype!(isize, ffi::RSMPI_INT64_T);

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
    pub fn contiguous<D>(count: Count, oldtype: &D) -> UserDatatype
    where
        D: Datatype,
    {
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
    pub fn vector<D>(count: Count, blocklength: Count, stride: Count, oldtype: &D) -> UserDatatype
    where
        D: Datatype,
    {
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
    pub fn heterogeneous_vector<D>(
        count: Count,
        blocklength: Count,
        stride: Address,
        oldtype: &D,
    ) -> UserDatatype
    where
        D: Datatype,
    {
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
    pub fn indexed<D>(blocklengths: &[Count], displacements: &[Count], oldtype: &D) -> UserDatatype
    where
        D: Datatype,
    {
        assert_eq!(blocklengths.len(), displacements.len());
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_indexed(
                blocklengths.count(),
                blocklengths.as_ptr(),
                displacements.as_ptr(),
                oldtype.as_raw(),
                &mut newtype,
            );
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
    pub fn heterogeneous_indexed<D>(
        blocklengths: &[Count],
        displacements: &[Address],
        oldtype: &D,
    ) -> UserDatatype
    where
        D: Datatype,
    {
        assert_eq!(blocklengths.len(), displacements.len());
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_create_hindexed(
                blocklengths.count(),
                blocklengths.as_ptr(),
                displacements.as_ptr(),
                oldtype.as_raw(),
                &mut newtype,
            );
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Construct a new type out of blocks of the same length and individual displacements.
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn indexed_block<D>(
        blocklength: Count,
        displacements: &[Count],
        oldtype: &D,
    ) -> UserDatatype
    where
        D: Datatype,
    {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_create_indexed_block(
                displacements.count(),
                blocklength,
                displacements.as_ptr(),
                oldtype.as_raw(),
                &mut newtype,
            );
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
    pub fn heterogeneous_indexed_block<D>(
        blocklength: Count,
        displacements: &[Address],
        oldtype: &D,
    ) -> UserDatatype
    where
        D: Datatype,
    {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        unsafe {
            ffi::MPI_Type_create_hindexed_block(
                displacements.count(),
                blocklength,
                displacements.as_ptr(),
                oldtype.as_raw(),
                &mut newtype,
            );
            ffi::MPI_Type_commit(&mut newtype);
        }
        UserDatatype(newtype)
    }

    /// Constructs a new datatype out of blocks of different length, displacement and datatypes
    ///
    /// # Examples
    /// See `examples/structured.rs`
    ///
    /// # Standard section(s)
    ///
    /// 4.1.2
    pub fn structured(
        count: Count,
        blocklengths: &[Count],
        displacements: &[Address],
        types: &[&Datatype<Raw = MPI_Datatype>],
    ) -> UserDatatype {
        let mut newtype: MPI_Datatype = unsafe { mem::uninitialized() };
        let types = types.iter().map(|t| t.as_raw()).collect::<Vec<_>>();
        unsafe {
            ffi::MPI_Type_create_struct(
                count,
                blocklengths.as_ptr(),
                displacements.as_ptr(),
                types.as_ptr(),
                &mut newtype,
            );
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
        assert_eq!(self.0, unsafe_extern_static!(ffi::RSMPI_DATATYPE_NULL));
    }
}

unsafe impl AsRaw for UserDatatype {
    type Raw = MPI_Datatype;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Datatype for UserDatatype {}

/// A Datatype describes the layout of messages in memory.
pub trait Datatype: AsRaw<Raw = MPI_Datatype> {}
impl<'a, D> Datatype for &'a D
where
    D: 'a + Datatype,
{
}

/// Something that has an associated datatype
pub unsafe trait AsDatatype {
    /// The type of the associated MPI datatype (e.g. `SystemDatatype` or `UserDatatype`)
    type Out: Datatype;
    /// The associated MPI datatype
    fn as_datatype(&self) -> Self::Out;
}

unsafe impl<T> AsDatatype for T
where
    T: Equivalence,
{
    type Out = <T as Equivalence>::Out;
    fn as_datatype(&self) -> Self::Out {
        <T as Equivalence>::equivalent_datatype()
    }
}

unsafe impl<T> AsDatatype for [T]
where
    T: Equivalence,
{
    type Out = <T as Equivalence>::Out;
    fn as_datatype(&self) -> Self::Out {
        <T as Equivalence>::equivalent_datatype()
    }
}

/// A countable collection of things.
pub unsafe trait Collection {
    /// How many things are in this collection.
    fn count(&self) -> Count;
}

unsafe impl<T> Collection for T
where
    T: Equivalence,
{
    fn count(&self) -> Count {
        1
    }
}

unsafe impl<T> Collection for [T]
where
    T: Equivalence,
{
    fn count(&self) -> Count {
        self.len()
            .value_as()
            .expect("Length of slice cannot be expressed as an MPI Count.")
    }
}

/// Provides a pointer to the starting address in memory.
pub unsafe trait Pointer {
    /// A pointer to the starting address in memory
    unsafe fn pointer(&self) -> *const c_void;
}

unsafe impl<T> Pointer for T
where
    T: Equivalence,
{
    unsafe fn pointer(&self) -> *const c_void {
        let p: *const T = self;
        p as *const c_void
    }
}

unsafe impl<T> Pointer for [T]
where
    T: Equivalence,
{
    unsafe fn pointer(&self) -> *const c_void {
        mem::transmute(self.as_ptr())
    }
}

/// Provides a mutable pointer to the starting address in memory.
pub unsafe trait PointerMut {
    /// A mutable pointer to the starting address in memory
    unsafe fn pointer_mut(&mut self) -> *mut c_void;
}

unsafe impl<T> PointerMut for T
where
    T: Equivalence,
{
    unsafe fn pointer_mut(&mut self) -> *mut c_void {
        let p: *mut T = self;
        p as *mut c_void
    }
}

unsafe impl<T> PointerMut for [T]
where
    T: Equivalence,
{
    unsafe fn pointer_mut(&mut self) -> *mut c_void {
        mem::transmute(self.as_mut_ptr())
    }
}

/// A buffer is a region in memory that starts at `pointer()` and contains `count()` copies of
/// `as_datatype()`.
pub unsafe trait Buffer: Pointer + Collection + AsDatatype {}
unsafe impl<T> Buffer for T
where
    T: Equivalence,
{
}
unsafe impl<T> Buffer for [T]
where
    T: Equivalence,
{
}

/// A mutable buffer is a region in memory that starts at `pointer_mut()` and contains `count()`
/// copies of `as_datatype()`.
pub unsafe trait BufferMut: PointerMut + Collection + AsDatatype {}
unsafe impl<T> BufferMut for T
where
    T: Equivalence,
{
}
unsafe impl<T> BufferMut for [T]
where
    T: Equivalence,
{
}

/// An immutable dynamically-typed buffer.
///
/// The buffer has a definite length and MPI datatype, but it is not yet known which Rust type it
/// corresponds to.  This is the MPI analogue of `&Any`.  It is semantically equivalent to the trait
/// object reference `&Buffer`.
#[derive(Copy, Clone, Debug)]
pub struct DynBuffer<'a> {
    ptr: *const c_void,
    len: Count,
    datatype: DatatypeRef<'a>,
}

unsafe impl<'a> Send for DynBuffer<'a> {}

unsafe impl<'a> Sync for DynBuffer<'a> {}

unsafe impl<'a> Collection for DynBuffer<'a> {
    fn count(&self) -> Count {
        self.len
    }
}

unsafe impl<'a> Pointer for DynBuffer<'a> {
    unsafe fn pointer(&self) -> *const c_void {
        self.ptr
    }
}

unsafe impl<'a> AsDatatype for DynBuffer<'a> {
    type Out = DatatypeRef<'a>;
    fn as_datatype(&self) -> Self::Out {
        self.datatype
    }
}

unsafe impl<'a> Buffer for DynBuffer<'a> {}

impl<'a> DynBuffer<'a> {
    /// Creates a buffer from a slice with whose type has an MPI equivalent.
    pub fn new<T: Equivalence>(buf: &'a [T]) -> Self {
        unsafe {
            let datatype = DatatypeRef::from_raw(T::equivalent_datatype().as_raw());
            Self::from_raw(buf.as_ptr(), buf.count(), datatype)
        }
    }

    /// Tests whether the buffer type matches `T`.
    pub fn is<T: Equivalence>(&self) -> bool {
        self.as_datatype().as_raw() == T::equivalent_datatype().as_raw()
    }

    /// Returns some slice if the type matches `T`, or `None` if it doesn't.
    pub fn downcast<T: Equivalence>(self) -> Option<&'a [T]> {
        if self.is::<T>() {
            unsafe { Some(slice::from_raw_parts(self.as_ptr() as _, self.len())) }
        } else {
            None
        }
    }

    /// Creates a buffer from its raw components.  The buffer must remain valid for `'a` and the
    /// pointer must not be null.
    pub unsafe fn from_raw<T>(ptr: *const T, len: Count, datatype: DatatypeRef<'a>) -> Self {
        debug_assert!(!ptr.is_null());
        Self {
            ptr: ptr as _,
            len,
            datatype,
        }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.count()
            .value_as()
            .expect("Length of DynBuffer cannot be expressed as a usize")
    }

    /// Returns `true` if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the underlying raw pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
}

/// A mutable dynamically-typed buffer.
///
/// The buffer has a definite length and MPI datatype, but it is not yet known which Rust type it
/// corresponds to.  This is the MPI analogue of `&mut Any`.  It is semantically equivalent to the
/// mutable trait object reference `&mut BufferMut`.
#[derive(Debug)]
pub struct DynBufferMut<'a> {
    ptr: *mut c_void,
    len: Count,
    datatype: DatatypeRef<'a>,
}

unsafe impl<'a> Send for DynBufferMut<'a> {}

unsafe impl<'a> Sync for DynBufferMut<'a> {}

unsafe impl<'a> Collection for DynBufferMut<'a> {
    fn count(&self) -> Count {
        self.len
    }
}

unsafe impl<'a> Pointer for DynBufferMut<'a> {
    unsafe fn pointer(&self) -> *const c_void {
        self.ptr
    }
}

unsafe impl<'a> PointerMut for DynBufferMut<'a> {
    unsafe fn pointer_mut(&mut self) -> *mut c_void {
        self.ptr
    }
}

unsafe impl<'a> Buffer for DynBufferMut<'a> {}

unsafe impl<'a> BufferMut for DynBufferMut<'a> {}

unsafe impl<'a> AsDatatype for DynBufferMut<'a> {
    type Out = DatatypeRef<'a>;
    fn as_datatype(&self) -> Self::Out {
        self.datatype
    }
}

impl<'a> DynBufferMut<'a> {
    /// Creates a mutable buffer from a mutable slice with whose type has an MPI equivalent.
    pub fn new<T: Equivalence>(buf: &'a mut [T]) -> Self {
        unsafe {
            let datatype = DatatypeRef::from_raw(T::equivalent_datatype().as_raw());
            Self::from_raw(buf.as_mut_ptr(), buf.count(), datatype)
        }
    }

    /// Tests whether the buffer type matches `T`.
    pub fn is<T: Equivalence>(&self) -> bool {
        self.as_datatype().as_raw() == T::equivalent_datatype().as_raw()
    }

    /// Returns some mutable slice if the type matches `T`, or `None` if it doesn't.
    pub fn downcast<T: Equivalence>(mut self) -> Option<&'a mut [T]> {
        if self.is::<T>() {
            unsafe {
                Some(slice::from_raw_parts_mut(
                    self.as_mut_ptr() as _,
                    self.len(),
                ))
            }
        } else {
            None
        }
    }

    /// Creates a buffer from its raw components.  The buffer must remain valid for `'a` and the
    /// pointer must not be null.
    pub unsafe fn from_raw<T>(ptr: *mut T, len: Count, datatype: DatatypeRef<'a>) -> Self {
        debug_assert!(!ptr.is_null());
        Self {
            ptr: ptr as _,
            len,
            datatype,
        }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.count()
            .value_as()
            .expect("Length of DynBufferMut cannot be expressed as a usize")
    }

    /// Returns `true` if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the underlying raw pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    /// Returns the underlying raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    /// Reborrows the buffer with a shorter lifetime.
    pub fn reborrow(&self) -> DynBuffer {
        unsafe { DynBuffer::from_raw(self.as_ptr(), self.count(), self.as_datatype()) }
    }

    /// Reborrows the buffer mutably with a shorter lifetime.
    pub fn reborrow_mut(&mut self) -> DynBufferMut {
        unsafe { DynBufferMut::from_raw(self.as_mut_ptr(), self.count(), self.as_datatype()) }
    }

    /// Makes the buffer immutable.
    pub fn downgrade(self) -> DynBuffer<'a> {
        unsafe { DynBuffer::from_raw(self.as_ptr(), self.count(), self.as_datatype()) }
    }
}

/// A buffer with a user specified count and datatype
///
/// # Safety
///
/// Views can be used to instruct the underlying MPI library to rummage around at arbitrary
/// locations in memory. This might be controlled later on using datatype bounds an slice lengths
/// but for now, all View constructors are marked `unsafe`.
pub struct View<'d, 'b, D, B: ?Sized>
where
    D: 'd + Datatype,
    B: 'b + Pointer,
{
    datatype: &'d D,
    count: Count,
    buffer: &'b B,
}

impl<'d, 'b, D, B: ?Sized> View<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + Pointer,
{
    /// Return a view of `buffer` containing `count` instances of MPI datatype `datatype`.
    ///
    /// # Examples
    /// See `examples/contiguous.rs`, `examples/vector.rs`
    pub unsafe fn with_count_and_datatype(
        buffer: &'b B,
        count: Count,
        datatype: &'d D,
    ) -> View<'d, 'b, D, B> {
        View {
            datatype,
            count,
            buffer,
        }
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> AsDatatype for View<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + Pointer,
{
    type Out = &'d D;
    fn as_datatype(&self) -> Self::Out {
        self.datatype
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> Collection for View<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + Pointer,
{
    fn count(&self) -> Count {
        self.count
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> Pointer for View<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + Pointer,
{
    unsafe fn pointer(&self) -> *const c_void {
        self.buffer.pointer()
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> Buffer for View<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + Pointer,
{
}

/// A buffer with a user specified count and datatype
///
/// # Safety
///
/// Views can be used to instruct the underlying MPI library to rummage around at arbitrary
/// locations in memory. This might be controlled later on using datatype bounds an slice lengths
/// but for now, all View constructors are marked `unsafe`.
pub struct MutView<'d, 'b, D, B: ?Sized>
where
    D: 'd + Datatype,
    B: 'b + PointerMut,
{
    datatype: &'d D,
    count: Count,
    buffer: &'b mut B,
}

impl<'d, 'b, D, B: ?Sized> MutView<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + PointerMut,
{
    /// Return a view of `buffer` containing `count` instances of MPI datatype `datatype`.
    ///
    /// # Examples
    /// See `examples/contiguous.rs`, `examples/vector.rs`
    pub unsafe fn with_count_and_datatype(
        buffer: &'b mut B,
        count: Count,
        datatype: &'d D,
    ) -> MutView<'d, 'b, D, B> {
        MutView {
            datatype,
            count,
            buffer,
        }
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> AsDatatype for MutView<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + PointerMut,
{
    type Out = &'d D;
    fn as_datatype(&self) -> Self::Out {
        self.datatype
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> Collection for MutView<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + PointerMut,
{
    fn count(&self) -> Count {
        self.count
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> PointerMut for MutView<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + PointerMut,
{
    unsafe fn pointer_mut(&mut self) -> *mut c_void {
        self.buffer.pointer_mut()
    }
}

unsafe impl<'d, 'b, D, B: ?Sized> BufferMut for MutView<'d, 'b, D, B>
where
    D: 'd + Datatype,
    B: 'b + PointerMut,
{
}

/// Describes how a `Buffer` is partitioned by specifying the count of elements and displacement
/// from the start of the buffer for each partition.
pub trait Partitioned {
    /// The count of elements in each partition.
    fn counts(&self) -> &[Count];
    /// The displacement from the start of the buffer for each partition.
    fn displs(&self) -> &[Count];

    /// A pointer to `counts()`
    unsafe fn counts_ptr(&self) -> *const Count {
        self.counts().as_ptr()
    }
    /// A pointer to `displs()`
    unsafe fn displs_ptr(&self) -> *const Count {
        self.displs().as_ptr()
    }
}

/// A buffer that is `Partitioned`
pub trait PartitionedBuffer: Partitioned + Pointer + AsDatatype {}

/// A mutable buffer that is `Partitioned`
pub trait PartitionedBufferMut: Partitioned + PointerMut + AsDatatype {}

/// Adds a partitioning to an existing `Buffer` so that it becomes `Partitioned`
pub struct Partition<'b, B: 'b + ?Sized, C, D> {
    buf: &'b B,
    counts: C,
    displs: D,
}

impl<'b, B: ?Sized, C, D> Partition<'b, B, C, D>
where
    B: 'b + Buffer,
    C: Borrow<[Count]>,
    D: Borrow<[Count]>,
{
    /// Partition `buf` using `counts` and `displs`
    pub fn new(buf: &B, counts: C, displs: D) -> Partition<B, C, D> {
        let n = buf.count();
        assert!(
            counts
                .borrow()
                .iter()
                .zip(displs.borrow().iter())
                .all(|(&c, &d)| c + d <= n)
        );

        Partition {
            buf,
            counts,
            displs,
        }
    }
}

unsafe impl<'b, B: ?Sized, C, D> AsDatatype for Partition<'b, B, C, D>
where
    B: 'b + AsDatatype,
{
    type Out = <B as AsDatatype>::Out;
    fn as_datatype(&self) -> Self::Out {
        self.buf.as_datatype()
    }
}

unsafe impl<'b, B: ?Sized, C, D> Pointer for Partition<'b, B, C, D>
where
    B: 'b + Pointer,
{
    unsafe fn pointer(&self) -> *const c_void {
        self.buf.pointer()
    }
}

impl<'b, B: ?Sized, C, D> Partitioned for Partition<'b, B, C, D>
where
    B: 'b,
    C: Borrow<[Count]>,
    D: Borrow<[Count]>,
{
    fn counts(&self) -> &[Count] {
        self.counts.borrow()
    }
    fn displs(&self) -> &[Count] {
        self.displs.borrow()
    }
}

impl<'b, B: ?Sized, C, D> PartitionedBuffer for Partition<'b, B, C, D>
where
    B: 'b + Pointer + AsDatatype,
    C: Borrow<[Count]>,
    D: Borrow<[Count]>,
{
}

/// Adds a partitioning to an existing `BufferMut` so that it becomes `Partitioned`
pub struct PartitionMut<'b, B: 'b + ?Sized, C, D> {
    buf: &'b mut B,
    counts: C,
    displs: D,
}

impl<'b, B: ?Sized, C, D> PartitionMut<'b, B, C, D>
where
    B: 'b + BufferMut,
    C: Borrow<[Count]>,
    D: Borrow<[Count]>,
{
    /// Partition `buf` using `counts` and `displs`
    pub fn new(buf: &mut B, counts: C, displs: D) -> PartitionMut<B, C, D> {
        let n = buf.count();
        assert!(
            counts
                .borrow()
                .iter()
                .zip(displs.borrow().iter())
                .all(|(&c, &d)| c + d <= n)
        );

        PartitionMut {
            buf,
            counts,
            displs,
        }
    }
}

unsafe impl<'b, B: ?Sized, C, D> AsDatatype for PartitionMut<'b, B, C, D>
where
    B: 'b + AsDatatype,
{
    type Out = <B as AsDatatype>::Out;
    fn as_datatype(&self) -> Self::Out {
        self.buf.as_datatype()
    }
}

unsafe impl<'b, B: ?Sized, C, D> PointerMut for PartitionMut<'b, B, C, D>
where
    B: 'b + PointerMut,
{
    unsafe fn pointer_mut(&mut self) -> *mut c_void {
        self.buf.pointer_mut()
    }
}

impl<'b, B: ?Sized, C, D> Partitioned for PartitionMut<'b, B, C, D>
where
    B: 'b,
    C: Borrow<[Count]>,
    D: Borrow<[Count]>,
{
    fn counts(&self) -> &[Count] {
        self.counts.borrow()
    }
    fn displs(&self) -> &[Count] {
        self.displs.borrow()
    }
}

impl<'b, B: ?Sized, C, D> PartitionedBufferMut for PartitionMut<'b, B, C, D>
where
    B: 'b + PointerMut + AsDatatype,
    C: Borrow<[Count]>,
    D: Borrow<[Count]>,
{
}

/// Returns the address of the argument in a format suitable for use with datatype constructors
///
/// # Examples
/// See `examples/structured.rs`
///
/// # Standard section(s)
///
/// 4.1.5
pub fn address_of<T>(x: &T) -> Address {
    let mut address = unsafe { mem::uninitialized() };
    let x: *const T = x;
    unsafe {
        ffi::MPI_Get_address(x as *const c_void, &mut address);
    }
    address
}
