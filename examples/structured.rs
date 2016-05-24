extern crate mpi;
#[macro_use] extern crate lazy_static;

use mpi::traits::*;
use mpi::datatype::UserDatatype;
use mpi::point_to_point as p2p;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct Message {
    a: f32,
    b: f64,

    c: i8,
    d: i16,
    e: i32,
    f: i64,

    g: u8,
    h: u16,
    i: u32,
    j: u64
}

unsafe impl Equivalence for Message {
    type Out = &'static UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        &*MESSAGE_DATATYPE
    }
}

lazy_static! {
    static ref MESSAGE_DATATYPE: UserDatatype = {
        use mpi::datatype::address_of;

        let msg = Message::default();
        let base = address_of(&msg);

        UserDatatype::structured(10,
            &[1, 1, 1, 1 ,1 ,1 ,1 ,1 ,1, 1],
            &[
                address_of(&msg.a) - base,
                address_of(&msg.b) - base,
                address_of(&msg.c) - base,
                address_of(&msg.d) - base,
                address_of(&msg.e) - base,
                address_of(&msg.f) - base,
                address_of(&msg.g) - base,
                address_of(&msg.h) - base,
                address_of(&msg.i) - base,
                address_of(&msg.j) - base
            ],
            &[
                &f32::equivalent_datatype(),
                &f64::equivalent_datatype(),
                &i8::equivalent_datatype(),
                &i16::equivalent_datatype(),
                &i32::equivalent_datatype(),
                &i64::equivalent_datatype(),
                &u8::equivalent_datatype(),
                &u16::equivalent_datatype(),
                &u32::equivalent_datatype(),
                &u64::equivalent_datatype()
            ]
        )
    };
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let next_process = world.process_at_rank(next_rank);
    let previous_rank = if rank - 1 >= 0 { rank - 1 } else { size - 1 };
    let previous_process = world.process_at_rank(previous_rank);

    let msg = Message { a: 1., b: 2., c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, i: 9, j: 10 };
    let mut buf = Message::default();

    let status = p2p::send_receive_into(&msg, &next_process, &mut buf, &previous_process);

    println!("received message: {:?}, with status: {:?}", buf, status);
    assert_eq!(msg, buf);
}
