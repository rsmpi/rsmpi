#![deny(warnings)]
#![allow(clippy::forget_copy)]

use std::mem::size_of;

use memoffset::offset_of;

use mpi::{
    datatype::{UncommittedUserDatatype, UserDatatype},
    traits::*,
    Address,
};

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct TupleType(
    [f32; 2],
    u8,
    u8,  // unused
    u16, // unused
);

// All subfields are `FromAnyBytes`.
unsafe impl EquivalenceFromAnyBytes for TupleType {}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct ComplexDatatype {
    b: mpi::Bool,
    _unused1: u8,
    _unused2: u16,
    ints: [i32; 4],
    tuple: TupleType,
}

unsafe impl Equivalence for ComplexDatatype {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1],
            &[
                offset_of!(ComplexDatatype, b) as Address,
                offset_of!(ComplexDatatype, ints) as Address,
                offset_of!(ComplexDatatype, tuple) as Address,
            ],
            &[
                bool::equivalent_datatype().into(),
                UncommittedUserDatatype::contiguous(4, &i32::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::structured(
                    &[2, 1],
                    &[
                        offset_of!(TupleType, 0) as Address,
                        offset_of!(TupleType, 1) as Address,
                    ],
                    &[f32::equivalent_datatype(), u8::equivalent_datatype()],
                )
                .as_ref(),
            ],
        )
    }
}

// All subfields are `FromAnyBytes`.
unsafe impl EquivalenceFromAnyBytes for ComplexDatatype {}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let root_process = world.process_at_rank(0);

    // verify that our types are the right size
    assert_eq!(12, size_of::<TupleType>());
    assert_eq!(32, size_of::<ComplexDatatype>());

    let mut data = if world.rank() == 0 {
        ComplexDatatype {
            b: true.into(),
            _unused1: 0,
            _unused2: 0,
            ints: [1, -2, 3, -4],
            tuple: TupleType(
                [-0.1, 0.1],
                7,
                0, // unused
                0, // unused
            ),
        }
    } else {
        ComplexDatatype::default()
    };

    root_process.broadcast_into(&mut data);

    assert!(data.b.valid().unwrap());
    assert_eq!([1, -2, 3, -4], data.ints);
    assert_eq!([-0.1, 0.1], data.tuple.0);
    assert_eq!(7, data.tuple.1);
}
