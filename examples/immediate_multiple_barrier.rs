/// Example using different types of requests with a request collection.
use mpi;
use mpi::traits::*;

#[cfg(msmpi)]
fn main() {}

const COUNT: usize = 128;

#[cfg(not(msmpi))]
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    // Try wait_any()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut finished = 0;
        while coll.incomplete() > 0 {
            coll.wait_any().unwrap();
            finished += 1;
            assert_eq!(coll.incomplete(), COUNT - finished);
        }
    });

    // Try wait_some()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut result = vec![];
        let mut finished = 0;
        while coll.incomplete() > 0 {
            coll.wait_some(&mut result);
            finished += result.len();
            assert_eq!(coll.incomplete(), COUNT - finished);
        }
    });

    // Try wait_all()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut result = vec![];
        coll.wait_all(&mut result);
        assert_eq!(result.len(), COUNT);
    });

    // Try test_any()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut finished = 0;
        while coll.incomplete() > 0 {
            if coll.test_any().is_some() {
                finished += 1;
            }
            assert_eq!(coll.incomplete(), COUNT - finished);
        }
        assert_eq!(finished, COUNT);
    });

    // Try test_some()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut result = vec![];
        let mut finished = 0;
        while coll.incomplete() > 0 {
            coll.test_some(&mut result);
            finished += result.len();
            assert_eq!(coll.incomplete(), COUNT - finished);
        }
        assert_eq!(finished, COUNT);
    });

    // Try test_all()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut result = vec![];
        while coll.incomplete() > 0 {
            if coll.test_all(&mut result) {
                assert_eq!(coll.incomplete(), 0);
            }
        }
        assert_eq!(result.len(), COUNT);
    });
}
