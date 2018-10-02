#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(missing_copy_implementations)]
#![cfg_attr(test, allow(trivial_casts))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy))]
include!(concat!(env!("OUT_DIR"), "/functions_and_types.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn mpi_fint_compiles() {
        if false {
            let _: RSMPI_Fint = unsafe { mem::uninitialized() };
        }
    }

    #[test]
    fn mpi_fint_comm_compiles() {
        if false {
            unsafe {
                let comm: MPI_Comm = mem::uninitialized();
                let fcomm: RSMPI_Fint = RSMPI_Comm_c2f(comm);
                let _: MPI_Comm = RSMPI_Comm_f2c(fcomm);
            }
        }
    }

    #[test]
    fn mpi_fint_errhandler_compiles() {
        if false {
            unsafe {
                let errhandler: MPI_Errhandler = mem::uninitialized();
                let ferrhandler: RSMPI_Fint = RSMPI_Errhandler_c2f(errhandler);
                let _: MPI_Errhandler = RSMPI_Errhandler_f2c(ferrhandler);
            }
        }
    }

    #[test]
    fn mpi_fint_file_compiles() {
        if false {
            unsafe {
                let file: MPI_File = mem::uninitialized();
                let ffile: RSMPI_Fint = RSMPI_File_c2f(file);
                let _: MPI_File = RSMPI_File_f2c(ffile);
            }
        }
    }

    #[test]
    fn mpi_fint_group_compiles() {
        if false {
            unsafe {
                let group: MPI_Group = mem::uninitialized();
                let fgroup: RSMPI_Fint = RSMPI_Group_c2f(group);
                let _: MPI_Group = RSMPI_Group_f2c(fgroup);
            }
        }
    }

    #[test]
    fn mpi_fint_info_compiles() {
        if false {
            unsafe {
                let info: MPI_Info = mem::uninitialized();
                let finfo: RSMPI_Fint = RSMPI_Info_c2f(info);
                let _: MPI_Info = RSMPI_Info_f2c(finfo);
            }
        }
    }

    #[test]
    fn mpi_fint_message_compiles() {
        if false {
            unsafe {
                let message: MPI_Message = mem::uninitialized();
                let fmessage: RSMPI_Fint = RSMPI_Message_c2f(message);
                let _: MPI_Message = RSMPI_Message_f2c(fmessage);
            }
        }
    }

    #[test]
    fn mpi_fint_op_compiles() {
        if false {
            unsafe {
                let op: MPI_Op = mem::uninitialized();
                let fop: RSMPI_Fint = RSMPI_Op_c2f(op);
                let _: MPI_Op = RSMPI_Op_f2c(fop);
            }
        }
    }

    #[test]
    fn mpi_fint_request_compiles() {
        if false {
            unsafe {
                let request: MPI_Request = mem::uninitialized();
                let frequest: RSMPI_Fint = RSMPI_Request_c2f(request);
                let _: MPI_Request = RSMPI_Request_f2c(frequest);
            }
        }
    }

    #[test]
    fn mpi_fint_datatype_compiles() {
        if false {
            unsafe {
                let datatype: MPI_Datatype = mem::uninitialized();
                let fdatatype: RSMPI_Fint = RSMPI_Type_c2f(datatype);
                let _: MPI_Datatype = RSMPI_Type_f2c(fdatatype);
            }
        }
    }

    #[test]
    fn mpi_fint_win_compiles() {
        if false {
            unsafe {
                let win: MPI_Win = mem::uninitialized();
                let fwin: RSMPI_Fint = RSMPI_Win_c2f(win);
                let _: MPI_Win = RSMPI_Win_f2c(fwin);
            }
        }
    }
}
