use std::{fmt::Debug, mem::MaybeUninit};

use mpi::{
    topology::{Communicator, SystemCommunicator},
    traits::*,
};

fn assert_equivalence<A, B>(comm: &SystemCommunicator, a: &A, b: &B)
where
    A: Buffer,
    B: BufferMut + PartialEq + Debug + Default,
{
    let packed = comm.pack(a);

    let mut new_b = B::default();
    comm.unpack_into(&packed, &mut new_b, 0);

    assert_eq!(b, &new_b);
}

fn main() {
    let universe = mpi::initialize().unwrap();

    let world = universe.world();

    #[derive(Equivalence)]
    struct MyProgramOpts {
        name: [u8; 100],
        num_cycles: u32,
        material_properties: [f64; 20],
    }

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct MyDataRust {
        b: mpi::Bool,
        f: f64,
        i: u16,
    }

    assert_equivalence(
        &world,
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
    );

    #[derive(Equivalence, Default, PartialEq, Debug)]
    #[repr(C)]
    struct MyDataC {
        b: mpi::Bool,
        f: f64,
        i: u16,
    }

    assert_equivalence(
        &world,
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
        &MyDataC {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
    );

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct MyDataOrdered {
        bf: (mpi::Bool, f64),
        i: u16,
    };

    assert_equivalence(
        &world,
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
        &MyDataOrdered {
            bf: (true.into(), 3.4),
            i: 7,
        },
    );

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct MyDataNestedTuple {
        bfi: (mpi::Bool, (f64, u16)),
    };

    assert_equivalence(
        &world,
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
        &MyDataNestedTuple {
            bfi: (true.into(), (3.4, 7)),
        },
    );

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct MyDataUnnamed(mpi::Bool, f64, u16);

    assert_equivalence(
        &world,
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
        &MyDataUnnamed(true.into(), 3.4, 7),
    );

    // `bool` is allowed here because we never receive into a value of type `BoolBoolBool`.
    #[derive(EquivalenceUnsafe, PartialEq, Debug)]
    struct BoolBoolBool(bool, bool, bool);

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct ThreeBool([mpi::Bool; 3]);

    assert_equivalence(
        &world,
        &BoolBoolBool(true, false, true),
        &ThreeBool([true.into(), false.into(), true.into()]),
    );

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct Empty;

    #[derive(Equivalence, PartialEq, Debug)]
    struct ZeroArray([i32; 0]);

    assert_equivalence(&world, &ZeroArray([]), &Empty);

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct Parent {
        b: mpi::Bool,
        child: Child,
    }

    #[derive(Equivalence, Default, PartialEq, Debug)]
    struct Child(f64, u16);

    assert_equivalence(
        &world,
        &MyDataRust {
            b: true.into(),
            f: 3.4,
            i: 7,
        },
        &Parent {
            b: true.into(),
            child: Child(3.4, 7),
        },
    );

    #[derive(EquivalenceUnsafe, Debug)]
    struct ComplexComplexComplex((i8, bool, i8), (i8, bool, i8), (i8, bool, i8));

    #[derive(EquivalenceUnsafe, Debug, PartialEq)]
    struct ThreeComplex([(i8, bool, i8); 3]);

    let a = ComplexComplexComplex((1, true, 1), (2, false, 2), (3, true, 3));

    let packed = world.pack(&a);

    let mut b: MaybeUninit<ThreeComplex> = MaybeUninit::uninit();
    world.unpack_into(&packed, &mut b, 0);

    // This is safe since we know that `ComplexComplexComplex` has a matching type-map to
    // `ThreeComplex`. In most HPC applications and environments, the risk of hitting UB related to
    // mis-matched types and the resulting type confusion is low, both in a practical and security
    // sense.
    let b = unsafe { b.assume_init() };

    assert_eq!(ThreeComplex([(1, true, 1), (2, false, 2), (3, true, 3)]), b);
}
