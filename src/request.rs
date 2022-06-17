//! Request objects for non-blocking operations
//!
//! Non-blocking operations such as `immediate_send()` return request objects that borrow any
//! buffers involved in the operation so as to ensure proper access restrictions. In order to
//! release the borrowed buffers from the request objects, a completion operation such as
//! [`wait()`](struct.Request.html#method.wait) or [`test()`](struct.Request.html#method.test) must
//! be used on the request object.
//!
//! **Note:** If the `Request` is dropped (as opposed to calling `wait` or `test` explicitly), the
//! program will panic.
//!
//! To enforce this rule, every request object must be registered to some pre-existing
//! [`Scope`](trait.Scope.html).  At the end of a `Scope`, all its remaining requests will be waited
//! for until completion.  Scopes can be created using either [`scope`](fn.scope.html) or
//! [`StaticScope`](struct.StaticScope.html).
//!
//! To handle request completion in an RAII style, a request can be wrapped in either
//! [`WaitGuard`](struct.WaitGuard.html) or [`CancelGuard`](struct.CancelGuard.html), which will
//! follow the respective policy for completing the operation.  When the guard is dropped, the
//! request will be automatically unregistered from its `Scope`.
//!
//! # Unfinished features
//!
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Test_cancelled()`

use std::cell::Cell;
use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::os::raw::c_int;

use crate::ffi;
use crate::ffi::{MPI_Request, MPI_Status};

use crate::point_to_point::Status;
use crate::raw::traits::*;
use crate::with_uninitialized;

/// Check if the request is `MPI_REQUEST_NULL`.
fn is_null(request: MPI_Request) -> bool {
    request == unsafe { ffi::RSMPI_REQUEST_NULL }
}

/// A request object for a non-blocking operation registered with a `Scope` of lifetime `'a`
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
pub struct Request<'a, D: ?Sized, S: Scope<'a> = StaticScope> {
    request: MPI_Request,
    data: &'a D,
    scope: S,
    phantom: PhantomData<Cell<&'a ()>>,
}

impl<'a, D: ?Sized, S: Scope<'a>> fmt::Debug for Request<'a, D, S>
where
    D: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.debug_struct("Request")
            .field("request", &self.request)
            .field("data", &self.data)
            .finish()
    }
}

unsafe impl<'a, D: ?Sized, S: Scope<'a>> AsRaw for Request<'a, D, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.request
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> Drop for Request<'a, D, S> {
    fn drop(&mut self) {
        panic!("request was dropped without being completed");
    }
}

/// Wait for the completion of one of the requests in the vector,
/// returns the index of the request completed and the status of the request.
///
/// The completed request is removed from the vector of requests.
///
/// If no Request is active None is returned.
///
/// # Examples
///
/// See `examples/wait_any.rs`
pub fn wait_any<'a, D, S: Scope<'a>>(
    requests: &mut Vec<Request<'a, D, S>>,
) -> Option<(usize, Status)> {
    let mut mpi_requests: Vec<_> = requests.iter().map(|r| r.as_raw()).collect();
    let mut index: i32 = mpi_sys::MPI_UNDEFINED;
    let size: i32 = mpi_requests
        .len()
        .try_into()
        .expect("Error while casting usize to i32");
    let status;
    unsafe {
        status = Status::from_raw(
            with_uninitialized(|s| {
                ffi::MPI_Waitany(size, mpi_requests.as_mut_ptr(), &mut index, s);
                s
            })
            .1,
        );
    }
    if index != mpi_sys::MPI_UNDEFINED {
        let u_index: usize = index.try_into().expect("Error while casting i32 to usize");
        assert!(is_null(mpi_requests[u_index]));
        let r = requests.remove(u_index);
        unsafe {
            r.into_raw();
        }
        Some((u_index, status))
    } else {
        None
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> Request<'a, D, S> {
    /// Construct a request object from the raw MPI type.
    ///
    /// # Requirements
    ///
    /// - The request is a valid, active request.  It must not be `MPI_REQUEST_NULL`.
    /// - The request must not be persistent.
    /// - All buffers associated with the request must outlive `'a`.
    /// - The request must not be registered with the given scope.
    ///
    /// # Safety
    /// - `request` must be a live MPI object.
    /// - `request` must not be used after calling `from_raw`.
    /// - Any buffers owned by `request` must live longer than `scope`.
    pub unsafe fn from_raw(
        request: MPI_Request,
        data: &'a D,
        scope: S,
    ) -> Self {
        debug_assert!(!is_null(request));
        scope.register();
        Self {
            request,
            data,
            scope,
            phantom: Default::default(),
        }
    }

    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    ///
    /// # Safety
    /// - The returned `MPI_Request` must be completed within the lifetime of the returned scope.
    pub unsafe fn into_raw(self) -> (MPI_Request, &'a D, S) {
        let request = ptr::read(&self.request);
        let data = ptr::read(&self.data);
        let scope = ptr::read(&self.scope);
        let _ = ptr::read(&self.phantom);
        mem::forget(self);
        scope.unregister();
        (request, data, scope)
    }

    /// Wait for the request to finish and unregister the request object from its scope.
    /// If provided, the status is written to the referent of the given reference.
    /// The referent `MPI_Status` object is never read. Also returns the data
    /// reference.
    fn wait_with(self, status: *mut MPI_Status) -> &'a D {
        unsafe {
            let (mut request, data, _) = self.into_raw();
            ffi::MPI_Wait(&mut request, status);
            assert!(is_null(request)); // persistent requests are not supported
            data
        }
    }

    /// Wait for the request to finish, unregister it, and return the data
    /// reference.
    pub fn wait_for_data(self) -> &'a D {
        // TODO: Just ignores the status for now, but this info might be needed.
        self.wait_with(unsafe { ffi::RSMPI_STATUS_IGNORE })
    }

    /// Wait for an operation to finish.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    pub fn wait(self) -> Status {
        unsafe { Status::from_raw(with_uninitialized(|status| self.wait_with(status)).1) }
    }

    /// Wait for an operation to finish, but don’t bother retrieving the `Status` information.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    pub fn wait_without_status(self) {
        self.wait_with(unsafe { ffi::RSMPI_STATUS_IGNORE });
    }

    /// Test whether an operation has finished.
    ///
    /// If the operation has finished, `Status` is returned.  Otherwise returns the unfinished
    /// `Request`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    pub fn test(self) -> Result<Status, Self> {
        unsafe {
            let mut status = MaybeUninit::uninit();
            let mut request = self.as_raw();

            let (_, flag) =
                with_uninitialized(|flag| ffi::MPI_Test(&mut request, flag, status.as_mut_ptr()));
            if flag != 0 {
                assert!(is_null(request)); // persistent requests are not supported
                let (_, _data, _) = self.into_raw();
                Ok(Status::from_raw(status.assume_init()))
            } else {
                Err(self)
            }
        }
    }

    /// Test whether an operation has finished.
    ///
    /// If the operation has finished, a tuple (`Status`, saved_data) is returned.
    /// Otherwise returns the unfinished `Request`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    pub fn test_with_data(self) -> Result<(Status, &'a D), Self> {
        unsafe {
            let mut status = MaybeUninit::uninit();
            let mut request = self.as_raw();

            let (_, flag) =
                with_uninitialized(|flag| ffi::MPI_Test(&mut request, flag, status.as_mut_ptr()));
            if flag != 0 {
                assert!(is_null(request)); // persistent requests are not supported
                let (_, data, _) = self.into_raw();
                Ok((Status::from_raw(status.assume_init()), data))
            } else {
                Err(self)
            }
        }
    }

    /// Initiate cancellation of the request.
    ///
    /// The MPI implementation is not guaranteed to fulfill this operation.  It may not even be
    /// valid for certain types of requests.  In the future, the MPI forum may [deprecate
    /// cancellation of sends][mpi26] entirely.
    ///
    /// [mpi26]: https://github.com/mpi-forum/mpi-issues/issues/26
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.8.4
    pub fn cancel(&self) {
        let mut request = self.as_raw();
        unsafe {
            ffi::MPI_Cancel(&mut request);
        }
    }

    /// Reduce the scope of a request.
    pub fn shrink_scope_to<'b, S2>(self, scope: S2) -> Request<'b, D, S2>
    where
        'a: 'b,
        S2: Scope<'b>,
    {
        unsafe {
            let (request, data, _) = self.into_raw();
            Request::from_raw(request, data, scope)
        }
    }
}

/// Guard object that waits for the completion of an operation when it is dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct WaitGuard<'a, D: ?Sized, S: Scope<'a> = StaticScope>(Option<Request<'a, D, S>>);

impl<'a, D: ?Sized, S: Scope<'a>> Drop for WaitGuard<'a, D, S> {
    fn drop(&mut self) {
        self.0.take().expect("invalid WaitGuard").wait();
    }
}

unsafe impl<'a, D: ?Sized, S: Scope<'a>> AsRaw for WaitGuard<'a, D, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_ref().expect("invalid WaitGuard").as_raw()
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> From<WaitGuard<'a, D, S>> for Request<'a, D, S> {
    fn from(mut guard: WaitGuard<'a, D, S>) -> Self {
        guard.0.take().expect("invalid WaitGuard")
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> From<Request<'a, D, S>> for WaitGuard<'a, D, S> {
    fn from(req: Request<'a, D, S>) -> Self {
        WaitGuard(Some(req))
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> WaitGuard<'a, D, S> {
    fn cancel(&self) {
        if let Some(ref req) = self.0 {
            req.cancel();
        }
    }
}

/// Guard object that tries to cancel and waits for the completion of an operation when it is
/// dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct CancelGuard<'a, D: ?Sized, S: Scope<'a> = StaticScope>(WaitGuard<'a, D, S>);

impl<'a, D: ?Sized, S: Scope<'a>> Drop for CancelGuard<'a, D, S> {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> From<CancelGuard<'a, D, S>> for WaitGuard<'a, D, S> {
    fn from(guard: CancelGuard<'a, D, S>) -> Self {
        unsafe {
            let inner = ptr::read(&guard.0);
            mem::forget(guard);
            inner
        }
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> From<WaitGuard<'a, D, S>> for CancelGuard<'a, D, S> {
    fn from(guard: WaitGuard<'a, D, S>) -> Self {
        CancelGuard(guard)
    }
}

impl<'a, D: ?Sized, S: Scope<'a>> From<Request<'a, D, S>> for CancelGuard<'a, D, S> {
    fn from(req: Request<'a, D, S>) -> Self {
        CancelGuard(WaitGuard::from(req))
    }
}

/// A common interface for [`LocalScope`](struct.LocalScope.html) and
/// [`StaticScope`](struct.StaticScope.html) used internally by the `request` module.
///
/// This trait is an implementation detail.  You shouldn’t have to use or implement this trait.
pub unsafe trait Scope<'a> {
    /// Registers a request with the scope.
    fn register(&self);

    /// Unregisters a request from the scope.
    ///
    /// # Safety
    /// DO NOT IMPLEMENT
    unsafe fn unregister(&self);
}

/// The scope that lasts as long as the entire execution of the program
///
/// Unlike `LocalScope<'a>`, `StaticScope` does not require any bookkeeping on the requests as every
/// request associated with a `StaticScope` can live as long as they please.
///
/// A `StaticScope` can be created simply by calling the `StaticScope` constructor.
///
/// # Invariant
///
/// For any `Request` registered with a `StaticScope`, its associated buffers must be `'static`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StaticScope;

unsafe impl Scope<'static> for StaticScope {
    fn register(&self) {}

    unsafe fn unregister(&self) {}
}

/// A temporary scope that lasts no more than the lifetime `'a`
///
/// Use `LocalScope` for to perform requests with temporary buffers.
///
/// To obtain a `LocalScope`, use the [`scope`](fn.scope.html) function.
///
/// # Invariant
///
/// For any `Request` registered with a `LocalScope<'a>`, its associated buffers must outlive `'a`.
///
/// # Panics
///
/// When `LocalScope` is dropped, it will panic if there are any lingering `Requests` that have not
/// yet been completed.
#[derive(Debug)]
pub struct LocalScope<'a> {
    num_requests: Cell<usize>,
    phantom: PhantomData<Cell<&'a ()>>, // Cell needed to ensure 'a is invariant
}

#[cold]
fn abort_on_unhandled_request() {
    let _ = std::panic::catch_unwind(|| {
        panic!("at least one request was dropped without being completed");
    });

    // There's no way to tell MPI to release the buffers that were passed to it. Therefore
    // we must abort execution.
    std::process::abort();
}

impl<'a> Drop for LocalScope<'a> {
    fn drop(&mut self) {
        if self.num_requests.get() != 0 {
            abort_on_unhandled_request();
        }
    }
}

unsafe impl<'a, 'b> Scope<'a> for &'b LocalScope<'a> {
    fn register(&self) {
        self.num_requests.set(self.num_requests.get() + 1)
    }

    unsafe fn unregister(&self) {
        self.num_requests.set(
            self.num_requests
                .get()
                .checked_sub(1)
                .expect("unregister has been called more times than register"),
        )
    }
}

/// Used to create a [`LocalScope`](struct.LocalScope.html)
///
/// The function creates a `LocalScope` and then passes it into the given
/// closure as an argument.
///
/// For safety reasons, all variables and buffers associated with a request
/// must exist *outside* the scope with which the request is registered.
///
/// It is typically used like this:
///
/// ```
/// /* declare variables and buffers here ... */
/// mpi::request::scope(|scope| {
///     /* perform sends and/or receives using 'scope' */
/// });
/// /* at end of scope, panic if there are requests that have not yet completed */
/// ```
///
/// # Examples
///
/// See `examples/immediate.rs`
pub fn scope<'a, F, R>(f: F) -> R
where
    F: FnOnce(&LocalScope<'a>) -> R,
{
    f(&LocalScope {
        num_requests: Default::default(),
        phantom: Default::default(),
    })
}

/// Create a scope for handling multiple request completion (such as wait_any(),
/// test_any(), test_some(), etc.). This takes a reserve amount indicating the
/// estimated amount of requests and a closure which will be called and passed a
/// (scope, RequestCollection).
///
/// Note: Both the RequestCollection and the scope will panic on drop if not all
/// requests have completed. Care must be taken to ensure that all requests have
/// completed. See the `incomplete()` method for checking the number of
/// outstanding requests.
pub fn multiple_scope<'a, F, R, D>(reserve: usize, f: F) -> R
where
    D: 'a + ?Sized,
    F: FnOnce(&LocalScope<'a>, &mut RequestCollection<'a, D>) -> R,
{
    f(
        &LocalScope {
            num_requests: Default::default(),
            phantom: Default::default(),
        },
        &mut RequestCollection::new(reserve),
    )
}

/// Request collection for managing multiple requests at the same time.
pub struct RequestCollection<'a, D: ?Sized> {
    /// Array of requests
    requests: Vec<MPI_Request>,
    /// List of data buffers attached to each request
    data: Vec<Option<&'a D>>,
    /// Request statuses
    statuses: Vec<MaybeUninit<MPI_Status>>,
    /// Pre-allocated indices buffer for use with testsome(), waitsome(), etc.
    indices: Vec<c_int>,
}

impl<'a, D: ?Sized> RequestCollection<'a, D> {
    /// Create a new RequestBuffer with a reserved size.
    fn new(reserve: usize) -> RequestCollection<'a, D> {
        let mut requests = vec![];
        let mut data = vec![];
        let mut statuses = vec![];
        let mut indices = vec![];
        requests.reserve(reserve);
        data.reserve(reserve);
        statuses.reserve(reserve);
        indices.reserve(reserve);
        RequestCollection {
            requests,
            data,
            statuses,
            indices,
        }
    }


    /// Return the total number of requests that are incomplete.
    pub fn incomplete(&self) -> usize {
        self.data.iter().map(|data| if data.is_some() { 1 } else { 0 }).sum()
    }

    /// Add the request to the collection. This unregisters the request from the
    /// scope. The collection then ensures that the request has completed.
    pub fn add<S>(&mut self, req: Request<'a, D, S>) -> usize
    where
        S: Scope<'a>,
    {
        let i = self.requests.len();
        let (req, data, _) = unsafe { req.into_raw() };
        self.requests.push(req);
        self.data.push(Some(data));
        self.statuses.push(MaybeUninit::<MPI_Status>::uninit());
        self.indices.push(0);
        i
    }

    /// Wait for any request to complete, and return an option containing
    /// (request_index, status, saved_data).
    pub fn wait_any(&mut self) -> Option<(usize, Status, &'a D)> {
         let mut i: c_int = 0;
         let (_res, status) = unsafe {
             let count = self.requests.len() as c_int;
             with_uninitialized(|status| {
                 ffi::MPI_Waitany(count, self.requests.as_mut_ptr(), &mut i, status)
             })
         };

         let i = i as usize;
         assert!(is_null(self.requests[i]));
         if let Some(data) = self.data[i].take() {
             Some((i, Status::from_raw(status), data))
         } else {
             None
         }
    }

    /// Wait for some of the requests to complete, fill result with references
    /// to the (request_index, status, saved_data) for each completed request
    /// and return the total number of completed requests.
    pub fn wait_some(&mut self, result: &mut [Option<(usize, Status, &'a D)>]) -> usize {
        assert_eq!(result.len(), self.requests.len());
        let mut count = 0;
        unsafe {
            let n = self.requests.len() as c_int;
            // NOTE: not using the return value here
            ffi::MPI_Waitsome(
                n,
                self.requests.as_mut_ptr(),
                &mut count,
                self.indices.as_mut_ptr(),
                self.statuses.as_mut_ptr() as *mut MPI_Status,
            );
        };
        let count = count as usize;
        for (i, elm) in result.iter_mut().enumerate() {
            elm.take();
            if i < count {
                let idx = self.indices[i] as usize;
                // Persistent requests check
                assert!(is_null(self.requests[idx]));
                if let Some(data) = self.data[idx].take() {
                    let status = unsafe { self.statuses[idx].assume_init() };
                    let status = Status::from_raw(status);
                    let _ = elm.insert((idx, status, data));
                }
            }
        }
        count
    }

    /// Wait for all requests to complete, putting (request_index, status, saved_data)
    /// into result for every completed request.
    pub fn wait_all(&mut self, result: &mut [Option<(usize, Status, &'a D)>]) {
        assert_eq!(result.len(), self.requests.len());
        let _res = unsafe {
            ffi::MPI_Waitall(
                self.requests.len().try_into().unwrap(),
                self.requests.as_mut_ptr(),
                self.statuses.as_mut_ptr() as *mut MPI_Status,
            )
        };
        for (i, elm) in result.iter_mut().enumerate() {
            // Ensure there are no persistent requests
            assert!(is_null(self.requests[i]));
            elm.take();
            if let Some(data) = self.data[i].take() {
                let status = unsafe { self.statuses[i].assume_init() };
                let status = Status::from_raw(status);
                let _ = elm.insert((i, status, data));
            }
        }
    }

    /// Test for the completion of any requests. Returns an option containing
    /// (request_index, status, saved_data).
    pub fn test_any(&mut self) -> Option<(usize, Status, &'a D)> {
        let n = self.requests.len() as c_int;
        let mut i = 0;
        let mut flag = 0;
        let (_, status) = unsafe {
            with_uninitialized(|status| {
                ffi::MPI_Testany(
                    n,
                    self.requests.as_mut_ptr(),
                    &mut i,
                    &mut flag,
                    status,
                )
            })
        };

        if flag != 0 {
            let i = i as usize;
            assert!(is_null(self.requests[i]));
            if let Some(data) = self.data[i].take() {
                Some((i, Status::from_raw(status), data))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Test for the completion of some requests. Completed request data will be
    /// stored in the result buffer in a tuple (request_index, status, saved_data).
    /// None values can be ignored. The result buffer must have a length equal to
    /// the number of requests added to this collection (the number of times `add()`
    /// was called.
    ///
    /// Returns the number of requests completed.
    pub fn test_some(&mut self, result: &mut [Option<(usize, Status, &'a D)>]) -> usize {
        assert_eq!(result.len(), self.requests.len());
        let n = self.requests.len() as c_int;
        let mut count = 0;
        unsafe {
            ffi::MPI_Testsome(
                n,
                self.requests.as_mut_ptr(),
                &mut count,
                self.indices.as_mut_ptr(),
                self.statuses.as_mut_ptr() as *mut MPI_Status,
            );
        }
        let count = count as usize;
        for (i, elm) in result.iter_mut().enumerate() {
            elm.take();
            if i < count {
                let idx = self.indices[i] as usize;
                assert!(is_null(self.requests[i]));
                let status = unsafe { self.statuses[i].assume_init() };
                if let Some(data) = self.data[idx].take() {
                    let _ = elm.insert((idx, Status::from_raw(status), data));
                }
            }
        }
        count
    }

    /// Test for the completion of all requests. Saved data used by the
    /// completed requests is stored in the result buffer. The result buffer
    /// must have a length equivalent to the number of stored requests,
    /// including those that have previously completed.
    pub fn test_all(&mut self, result: &mut [Option<(usize, Status, &'a D)>]) -> bool {
        assert_eq!(result.len(), self.requests.len());
        let n = self.requests.len() as c_int;
        let mut flag = 0;
        unsafe {
            ffi::MPI_Testall(
                n,
                self.requests.as_mut_ptr(),
                &mut flag,
                self.statuses.as_mut_ptr() as *mut MPI_Status,
            );
        }
        if flag != 0 {
            for (i, elm) in result.iter_mut().enumerate() {
                assert!(is_null(self.requests[i]));
                elm.take();
                let status = unsafe { self.statuses[i].assume_init() };
                if let Some(data) = self.data[i].take() {
                    let _ = elm.insert((i, Status::from_raw(status), data));
                }
                // self.finish_request(i);
            }
            true
        } else {
            false
        }
    }
}

/// Drop implementation to ensure that all requests have actually completed.
impl<'a, D: ?Sized> Drop for RequestCollection<'a, D> {
    fn drop(&mut self) {
        if !self.data.iter().all(|c| c.is_none()) {
            panic!("some requests have not completed");
        }
    }
}
