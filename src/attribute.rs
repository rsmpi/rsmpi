//! Attribute caching on communicators

use std::{any::TypeId, collections::HashMap, ffi::c_void, os::raw::c_int, ptr, sync::RwLock};

use once_cell::sync::Lazy;

use crate::{ffi, traits::AsRaw};

/// Topology traits
pub mod traits {
    pub use super::CommAttribute;
}

pub(crate) static COMM_ATTRS: Lazy<RwLock<HashMap<TypeId, AttributeKey>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Attributes are user data that can be owned by communicators and accessed by
/// users. They are useful when libraries pass communicators to a different
/// library and get it back in a callback.
///
/// # Standard section(s)
///
/// 7.7.1
pub trait CommAttribute
where
    Self: 'static + Sized + Clone,
{
    /// When a communicator is duplicated, attributes can either be cloned to
    /// the new communicator or not propagated at all. Implementations of
    /// CommAttribute can determine this behavior by defining this associated
    /// constant. The default does not propagated attributes to cloned
    /// communicators.
    const CLONE_ON_DUP: bool = false;

    /// Callback invoked by `MPI Comm_free`, `MPI_Comm_disconnect`, and
    /// `MPI_Comm_delete_attr` to delete an attribute.
    ///
    /// User-defined attributes should not need to override this default
    /// implementation.
    ///
    /// # Safety
    ///
    /// This default implementation goes with the boxing in
    /// `AnyCommunicator::set_attr()`.
    unsafe extern "C" fn comm_delete_attr_fn(
        _comm: ffi::MPI_Comm,
        _key: c_int,
        val: *mut c_void,
        _extra_state: *mut c_void,
    ) -> c_int {
        let _to_drop = Box::from_raw(val as *mut Self);
        ffi::MPI_SUCCESS as i32
    }

    /// Callback invoked by `MPI_Comm_dup()`, `MPI_Comm_idup()`, and variants to
    /// (optionally) clone the attribute to the new communicator. The behavior
    /// of this function is determined by `Self::CLONE_ON_DUP`, which should be
    /// sufficient to obtain desired semantics.
    ///
    /// User-defined attributes should not need to override this default
    /// implementation.
    ///
    /// # Safety
    ///
    /// This default implementation must only be used with boxed attributes, as
    /// in `AnyCommunicator::set_attr()`.
    unsafe extern "C" fn comm_copy_attr_fn(
        _old_comm: ffi::MPI_Comm,
        _key: c_int,
        _extra_state: *mut c_void,
        val_in: *mut c_void,
        val_out: *mut c_void,
        flag: *mut c_int,
    ) -> c_int {
        if Self::CLONE_ON_DUP {
            let b_in = Box::from_raw(val_in as *mut Self);
            let b_out = b_in.clone();
            *(val_out as *mut *mut Self) = Box::into_raw(b_out);
            Box::into_raw(b_in); // deconstruct to avoid dropping
            *flag = 1;
        } else {
            *flag = 0;
        }
        ffi::MPI_SUCCESS as i32
    }

    /// Get the attribute key for this attribute. User keys are provisioned by
    /// `MPI_Comm_create_keyval()`, which we store in the universe and free
    /// prior to `MPI_Finalize()` when the universe is dropped.
    ///
    /// In multi-language projects for which `MPI_Comm_create_keyval()` is
    /// called from a different language, the attribute can be set and retrieved
    /// in Rust by overriding this default implementation to return the foreign
    /// key.
    fn get_key() -> AttributeKey {
        let id = TypeId::of::<Self>();
        {
            let comm_attrs = COMM_ATTRS.read().expect("COMM_ATTRS RwLock poisoned");
            if let Some(key) = comm_attrs.get(&id) {
                return key.clone();
            }
        }
        let mut key: i32 = 0;
        unsafe {
            ffi::MPI_Comm_create_keyval(
                Some(Self::comm_copy_attr_fn),
                Some(Self::comm_delete_attr_fn),
                &mut key,
                ptr::null_mut(),
            );
        }
        let key = AttributeKey(key);
        let mut comm_attrs = COMM_ATTRS.write().expect("COMM_ATTRS RwLock poisoned");
        comm_attrs.insert(id, key.clone());
        key
    }
}

/// Attribute keys are used internally to access attributes. They are obtained
/// with the associated function `CommAttribute::get_key()`.
///
/// User keys are created with `MPI_Comm_create_keyval()` and should be freed
/// with `MPI_Comm_free_keyval()`. They are provisioned in the default
/// implementation of `CommAttribute::get_key()` and stored persistently in
/// `COMM_ATTRS`.
///
/// System keys are automatically available and are not passed to
/// `MPI_Comm_free_keyval()`.
#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
pub struct AttributeKey(i32);

impl AttributeKey {
    /// Create a new attribute key. This is mostly unnecessary for users, but
    /// may be used when implementing `CommAttribute::get_key()` to be used with
    /// a foreign keyval.
    ///
    /// # Safety
    ///
    /// The key must be a valid predefined system keyval or obtained by
    /// `MPI_Comm_create_keyval()`. Strictly speaking, I think this can be safe,
    /// but an invalid value will cause `MPI_Comm_get_attr()` and
    /// `MPI_Comm_set_attr()` to fail. Note that they can also be made to fail
    /// if `MPI_Comm_free_keyval()` is called after creating an `AttributeKey`
    /// and before it is used. I'm leaving it `unsafe` to be defensive and
    /// because the caller should think carefully about lifetime of their
    /// foreign keyvals when interoperating with Rust.
    pub unsafe fn new_unchecked(k: i32) -> Self {
        Self(k)
    }
}

unsafe impl AsRaw for AttributeKey {
    type Raw = c_int;
    fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

/// For obtaining the universe size attribute
#[repr(C)]
#[derive(Clone)]
pub(crate) struct UniverseSize(c_int);

impl CommAttribute for UniverseSize {
    fn get_key() -> AttributeKey {
        unsafe { AttributeKey::new_unchecked(ffi::MPI_UNIVERSE_SIZE as i32) }
    }
}

impl TryFrom<&UniverseSize> for usize {
    type Error = std::num::TryFromIntError;
    fn try_from(s: &UniverseSize) -> Result<Self, Self::Error> {
        usize::try_from(s.0)
    }
}

/// For obtaining the appnum attribute of MPI_COMM_WORLD
#[repr(C)]
#[derive(Clone)]
pub(crate) struct AppNum(c_int);

impl CommAttribute for AppNum {
    fn get_key() -> AttributeKey {
        unsafe { AttributeKey::new_unchecked(ffi::MPI_APPNUM as i32) }
    }
}

impl From<&AppNum> for isize {
    fn from(an: &AppNum) -> Self {
        an.0 as isize
    }
}
