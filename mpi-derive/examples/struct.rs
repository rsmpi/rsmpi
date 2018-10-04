extern crate mpi;
#[macro_use]
extern crate mpi_derive;

use mpi::{collective::Root, topology::Communicator};

fn main() {
    let universe = mpi::initialize().unwrap();

    let world = universe.world();

    let root_process = world.process_at_rank(0);

    #[derive(Datatype, Default)]
    struct MyDataRust {
        b: bool,
        f: f64,
        i: u16,
    }

    #[derive(Datatype, Default)]
    #[repr(C)]
    struct MyDataC {
        b: bool,
        f: f64,
        i: u16,
    }

    if world.rank() == 0 {
        let mut data = MyDataRust {
            b: true,
            f: 3.4,
            i: 7,
        };

        root_process.broadcast_into(&mut data);

        assert_eq!(true, data.b);
        assert_eq!(3.4, data.f);
        assert_eq!(7, data.i);
    } else {
        let mut data = MyDataC::default();
        root_process.broadcast_into(&mut data);

        assert_eq!(true, data.b);
        assert_eq!(3.4, data.f);
        assert_eq!(7, data.i);
    };
}
