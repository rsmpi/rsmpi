/// Example using different types of requests with a request collection.
use mpi;
use mpi::traits::*;

#[cfg(msmpi)]
fn main() {
}

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
        while coll.incomplete() > 0 {
            coll.wait_any().unwrap();
        }
    });

    // Try wait_some()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut result = vec![None; COUNT];
        while coll.incomplete() > 0 {
            let count = coll.wait_some(&mut result);
            let mut i = 0;
            for elm in result.iter() {
                if let Some(_) = elm {
                    i += 1;
                }
            }
            assert_eq!(i, count);
        }
    });

    // Try wait_all()
    mpi::request::multiple_scope(COUNT, |_, coll| {
        for _ in 0..COUNT {
            let req = world.immediate_barrier();
            coll.add(req);
        }
        let mut result = vec![None; COUNT ];
        coll.wait_all(&mut result);
        assert!(result.iter().all(|elm| elm.is_some()));
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
        let mut result = vec![None; COUNT];
        let mut finished = 0;
        while coll.incomplete() > 0 {
            let count = coll.test_some(&mut result);
            finished += count;
            if count > 0 {
                let mut i = 0;
                for elm in result.iter() {
                    if let Some(_) = elm {
                        i += 1;
                    }
                }
                assert_eq!(i, count);
            }
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
        let mut result = vec![None; COUNT];
        while coll.incomplete() > 0 {
            if coll.test_all(&mut result) {
                assert_eq!(coll.incomplete(), 0);
            }
        }
        assert!(result.iter().all(|elm| elm.is_some()));
    });
}
