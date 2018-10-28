use std::mem;

use conv::ConvUtil;

use super::super::{datatype::traits::*, ffi, raw::traits::*, Count, IntArray};
use super::{AsCommunicator, Communicator, IntoTopology, Rank, UserCommunicator};
use ffi::MPI_Comm;

/// Contains arrays describing the layout of the
/// [`CartesianCommunicator`](struct.CartesianCommunicator.html).
///
/// dims[i] is the extent of the array in axis i, periods[i] is true if axis i is periodic, and
/// coords[i] is the cartesian coordinate for the local rank in axis i.
///
/// Each array, when received from a method in
/// [`CartesianCommunicator`](struct.CartesianCommunicator.html), will be of length
/// [`num_dimensions`](struct.CartesianCommunicator.html#method.num_dimensions).
pub struct CartesianLayout {
    /// dims[i] is the extent of the array in axis i
    pub dims: Vec<Count>,
    /// periods[i] is true if axis i is periodic, meaning an element at dims[i] - 1 in axis i
    /// is neighbors with element 0 in axis i
    pub periods: Vec<bool>,
    /// coords[i] is the cartesian coordinate for the local rank in axis i
    pub coords: Vec<Count>,
}

/// A `CartesianCommunicator` is an MPI communicator object where ranks are laid out in an
/// n-dimensional cartesian space. This gives ranks neighbors in each of those dimensions, and MPI
/// is able to optimize the layout of these ranks to improve physical locality.
///
/// # Standard Section(s)
///
/// 7
pub struct CartesianCommunicator(pub(crate) UserCommunicator);

impl CartesianCommunicator {
    /// Given a valid `MPI_Comm` handle in `raw`, returns a `CartesianCommunicator` value if, and
    /// only if:
    /// - The handle is not `MPI_COMM_NULL`
    /// - The topology of the communicator is `MPI_CART`
    ///
    /// Otherwise returns None.
    ///
    /// Behavior is undefined if `raw` is not a valid `MPI_Comm` handle.
    ///
    /// # Parameters
    /// * `raw` - Handle to a valid `MPI_Comm` object
    pub unsafe fn from_raw(raw: MPI_Comm) -> Option<CartesianCommunicator> {
        UserCommunicator::from_raw(raw).and_then(|comm| match comm.into_topology() {
            IntoTopology::Cartesian(c) => Some(c),
            incorrect => {
                // Forget the comm object so it's not dropped
                mem::forget(incorrect);

                None
            }
        })
    }

    /// Creates a `CartesianCommunicator` from `raw`.
    ///
    /// Behavior is undefined if `raw`:
    /// * Is not a valid `MPI_Comm` value
    /// * Is equal to `MPI_COMM_NULL`
    /// * Does not have the `MPI_CART` topology
    ///
    /// # Parameters
    /// * `raw` - Handle to a valid `MPI_CART` `MPI_Comm` object
    pub unsafe fn from_raw_unchecked(raw: MPI_Comm) -> CartesianCommunicator {
        debug_assert_ne!(raw, ffi::RSMPI_COMM_NULL);
        CartesianCommunicator(UserCommunicator::from_raw_unchecked(raw))
    }

    /// Returns the number of dimensions that the Cartesian communicator was established over.
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cartdim_get)
    pub fn num_dimensions(&self) -> Count {
        unsafe {
            let mut count = mem::uninitialized();
            ffi::MPI_Cartdim_get(self.as_raw(), &mut count);
            count
        }
    }

    /// Returns the topological structure of the Cartesian communicator
    ///
    /// Behavior is undefined if `dims`, `periods`, and `coords` are not of length
    /// [`num_dimensions`](#method.num_dimensions).
    ///
    /// Prefer [`get_layout_into`](#method.get_layout_into)
    ///
    /// # Parameters
    /// * `dims` - array of spatial extents for the cartesian space
    /// * `periods` - Must match length of `dims`. `periods[i]` indicates if axis i is periodic.
    ///     i.e. if true, the element at `dims[i] - 1` in axis i is a neighbor of element 0 in axis
    ///     i
    /// * `coords` - `coords[i]` is the location offset in axis i of this rank
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_get)
    pub unsafe fn get_layout_into_unchecked(
        &self,
        dims: &mut [Count],
        periods: &mut [bool],
        coords: &mut [Count],
    ) {
        let mut periods_int: IntArray = smallvec![mem::uninitialized(); periods.len()];

        ffi::MPI_Cart_get(
            self.as_raw(),
            self.num_dimensions(),
            dims.as_mut_ptr(),
            periods_int.as_mut_ptr(),
            coords.as_mut_ptr(),
        );

        for (p, pi) in periods.iter_mut().zip(periods_int.iter()) {
            *p = match pi {
                0 => false,
                1 => true,
                _ => panic!(
                    "Received an invalid boolean value ({}) from the MPI implementation",
                    pi
                ),
            }
        }
    }

    /// Returns the topological structure of the Cartesian communicator
    ///
    /// Panics if `dims`, `periods`, and `coords` are not of length
    /// [`num_dimensions`](#method.num_dimensions).
    ///
    /// # Parameters
    /// * `dims` - array of spatial extents for the cartesian space
    /// * `periods` - Must match length of `dims`. `periods[i]` indicates if axis i is periodic.
    ///     i.e. if true, the element at `dims[i] - 1` in axis i is a neighbor of element 0 in axis
    ///     i
    /// * `coords` - `coords[i]` is the location offset in axis i of this rank
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_get)
    pub fn get_layout_into(&self, dims: &mut [Count], periods: &mut [bool], coords: &mut [Count]) {
        assert_eq!(
            dims.count(),
            periods.count(),
            "dims, periods, and coords must be the same length"
        );
        assert_eq!(
            dims.count(),
            coords.count(),
            "dims, periods, and coords must be the same length"
        );

        assert_eq!(
            self.num_dimensions(),
            dims.count(),
            "dims, periods, and coords must be equal in length to num_dimensions()"
        );

        unsafe { self.get_layout_into_unchecked(dims, periods, coords) }
    }

    /// Returns the topological structure of the Cartesian communicator
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_get)
    pub fn get_layout(&self) -> CartesianLayout {
        let num_dims = self
            .num_dimensions()
            .value_as()
            .expect("Received unexpected value from MPI_Cartdim_get");

        let mut layout = CartesianLayout {
            dims: vec![0; num_dims],
            periods: vec![false; num_dims],
            coords: vec![0; num_dims],
        };

        self.get_layout_into(
            &mut layout.dims[..],
            &mut layout.periods[..],
            &mut layout.coords[..],
        );

        layout
    }

    /// Converts a set of cartesian coordinates to its rank in the CartesianCommunicator.
    ///
    /// Behavior is undefined if `coords` is not of length
    /// [`num_dimensions`](#method.num_dimensions).
    ///
    /// Coordinates in periodic axes that are out of range are shifted back into the dimensions of
    /// the communiactor.
    ///
    /// Behavior is undefined if any coordinates in non-periodic axes are outside the dimensions of
    /// the communicator.
    ///
    /// Prefer [`coordinates_to_rank`](#method.coordinates_to_rank)
    ///
    /// # Parameters
    /// * `coords` - `coords[i]` is a location offset in axis i
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_rank)
    pub unsafe fn coordinates_to_rank_unchecked(&self, coords: &[Count]) -> Rank {
        let mut rank: Rank = mem::uninitialized();
        ffi::MPI_Cart_rank(self.as_raw(), coords.as_ptr(), &mut rank);
        rank
    }

    /// Converts a set of cartesian coordinates to its rank in the CartesianCommunicator.
    ///
    /// Panics if `coords` is not of length [`num_dimensions`](#method.num_dimensions).
    ///
    /// Coordinates in periodic axes that are out of range are shifted back into the dimensions of
    /// the communiactor.
    ///
    /// Panics if any coordinates in non-periodic axes are outside the dimensions of the
    /// communicator.
    ///
    /// # Parameters
    /// * `coords` - `coords[i]` is a location offset in axis i
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_rank)
    pub fn coordinates_to_rank(&self, coords: &[Count]) -> Rank {
        let num_dims = self
            .num_dimensions()
            .value_as()
            .expect("Received unexpected value from MPI_Cartdim_get");

        assert_eq!(
            num_dims,
            coords.len(),
            "The coordinates slice must be the same length as the number of dimension in the \
             CartesianCommunicator"
        );

        let layout = self.get_layout();

        for i in 0..num_dims {
            if !layout.periods[i] {
                assert!(
                    coords[i] > 0,
                    "The non-periodic coordinate (coords[{}] = {}) must be greater than 0.",
                    i,
                    coords[i]
                );
                assert!(
                    coords[i] <= layout.dims[i],
                    "The non-period coordinate (coords[{}] = {}) must be within the bounds of the \
                     CartesianCoordinator (dims[{}] = {})",
                    i,
                    coords[i],
                    i,
                    layout.dims[i]
                );
            }
        }

        unsafe { self.coordinates_to_rank_unchecked(coords) }
    }

    /// Receives into `coords` the cartesian coordinates of `rank`.
    ///
    /// Behavior is undefined if `rank` is not a non-negative value less than
    /// [`size`](trait.Communicator.html#method.size).
    ///
    /// Prefer [`rank_to_coordinates_into`](#method.rank_to_coordinates_into)
    ///
    /// # Parameters
    /// * `rank` - A rank in the communicator
    /// * `coords` - `coords[i]` is the cartesian coordinate of `rank` in axis i
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_coords)
    pub unsafe fn rank_to_coordinates_into_unchecked(&self, rank: Rank, coords: &mut [Count]) {
        ffi::MPI_Cart_coords(self.as_raw(), rank, coords.count(), coords.as_mut_ptr());
    }

    /// Receives into `coords` the cartesian coordinates of `rank`.
    ///
    /// Panics if `rank` is not a non-negative value less than
    /// [`size`](trait.Communicator.html#method.size).
    ///
    /// # Parameters
    /// * `rank` - A rank in the communicator
    /// * `coords` - `coords[i]` is the cartesian coordinate of `rank` in axis i
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_coords)
    pub fn rank_to_coordinates_into(&self, rank: Rank, coords: &mut [Count]) {
        assert!(
            rank >= 0 && rank < self.size(),
            "rank ({}) must be in the range [0,{})",
            rank,
            self.size()
        );

        unsafe { self.rank_to_coordinates_into_unchecked(rank, coords) }
    }

    /// Returns an array of `coords` with the cartesian coordinates of `rank`, where `coords[i]` is
    /// the cartesian coordinate of `rank` in axis i.
    ///
    /// Panics if `rank` is not a non-negative value less than
    /// [`size`](trait.Communicator.html#method.size).
    ///
    /// # Parameters
    /// * `rank` - A rank in the communicator
    ///
    /// # Standard section(s)
    /// 7.5.5 (MPI_Cart_coords)
    pub fn rank_to_coordinates(&self, rank: Rank) -> Vec<Count> {
        let mut coords = vec![
            0;
            self.num_dimensions()
                .value_as()
                .expect("Received unexpected value from MPI_Cartdim_get")
        ];

        self.rank_to_coordinates_into(rank, &mut coords[..]);

        coords
    }

    /// Retrieves targets in `dimension` shifted from the current rank by displacing in the negative
    /// direction by `displacement` units for the first returned rank and in the positive direction
    /// for the second returned rank.
    ///
    /// Behavior is undefined if `dimension` is not of length
    /// [`num_dimensions`](#method.num_dimensions).
    ///
    /// Prefer [`shift`](#method.shift)
    ///
    /// # Parameters
    /// * `dimension` - which axis to shift in
    /// * `displacement` - what offset to shift by in each direction
    ///
    /// # Standard section(s)
    /// 7.5.6 (MPI_Cart_shift)
    pub unsafe fn shift_unchecked(
        &self,
        dimension: Count,
        displacement: Count,
    ) -> (Option<Rank>, Option<Rank>) {
        let mut rank_source = mem::uninitialized();
        let mut rank_destination = mem::uninitialized();
        ffi::MPI_Cart_shift(
            self.as_raw(),
            dimension,
            displacement,
            &mut rank_source,
            &mut rank_destination,
        );

        let rank_source = if rank_source != ffi::RSMPI_PROC_NULL {
            Some(rank_source)
        } else {
            None
        };

        let rank_destination = if rank_destination != ffi::RSMPI_PROC_NULL {
            Some(rank_destination)
        } else {
            None
        };

        (rank_source, rank_destination)
    }

    /// Retrieves targets in `dimension` shifted from the current rank by displacing in the negative
    /// direction by `displacement` units for the first returned rank and in the positive direction
    /// for the second returned rank.
    ///
    /// Panics if `dimension` is not of length [`num_dimensions`](#method.num_dimensions).
    ///
    /// # Parameters
    /// * `dimension` - which axis to shift in
    /// * `displacement` - what offset to shift by in each direction
    ///
    /// # Standard section(s)
    /// 7.5.6 (MPI_Cart_shift)
    pub fn shift(&self, dimension: Count, displacement: Count) -> (Option<Rank>, Option<Rank>) {
        assert!(
            dimension >= 0,
            "dimension ({}) cannot be negative",
            dimension
        );

        assert!(
            dimension < self.num_dimensions(),
            "dimension ({}) is not valid for this communicator (num_dimensions = {})",
            dimension,
            self.num_dimensions(),
        );

        unsafe { self.shift_unchecked(dimension, displacement) }
    }

    /// Partitions an existing Cartesian communicator into a new Cartesian communicator in a lower
    /// dimension.
    ///
    /// Behavior is undefined if `retain` is not of length
    /// [`num_dimensions`](#method.num_dimensions).
    ///
    /// Prefer [`subgroup`](#method.subgroup)
    ///
    /// # Parameters
    /// * `retain` - if `retain[i]` is true, then axis i is retained in the new communicator
    ///
    /// # Standard section(s)
    /// 7.5.7 (MPI_Cart_sub)
    pub unsafe fn subgroup_unchecked(&self, retain: &[bool]) -> CartesianCommunicator {
        let retain_int: IntArray = retain.iter().map(|b| *b as _).collect();

        let mut newcomm = mem::uninitialized();
        ffi::MPI_Cart_sub(self.as_raw(), retain_int.as_ptr(), &mut newcomm);
        CartesianCommunicator::from_raw_unchecked(newcomm)
    }

    /// Partitions an existing Cartesian communicator into a new Cartesian communicator in a lower
    /// dimension.
    ///
    /// Panics if `retain` is not of length [`num_dimensions`](#method.num_dimensions).
    ///
    /// # Parameters
    /// * `retain` - if `retain[i]` is true, then axis i is retained in the new communicator
    ///
    /// # Standard section(s)
    /// 7.5.7 (MPI_Cart_sub)
    pub fn subgroup(&self, retain: &[bool]) -> CartesianCommunicator {
        assert_eq!(
            self.num_dimensions(),
            retain.count(),
            "The length of the retained dimensions array must be equal to the number of dimensions \
            in the CartesianCommunicator");

        unsafe { self.subgroup_unchecked(retain) }
    }
}

impl Communicator for CartesianCommunicator {}

impl AsCommunicator for CartesianCommunicator {
    type Out = CartesianCommunicator;
    fn as_communicator(&self) -> &Self::Out {
        self
    }
}

unsafe impl AsRaw for CartesianCommunicator {
    type Raw = MPI_Comm;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_raw()
    }
}
