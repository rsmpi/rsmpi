#ifndef RSMPI_INCLUDED
#define RSMPI_INCLUDED
#include "mpi.h"

extern const MPI_Datatype RSMPI_FLOAT;
extern const MPI_Datatype RSMPI_DOUBLE;

extern const MPI_Datatype RSMPI_INT8_T;
extern const MPI_Datatype RSMPI_INT16_T;
extern const MPI_Datatype RSMPI_INT32_T;
extern const MPI_Datatype RSMPI_INT64_T;

extern const MPI_Datatype RSMPI_UINT8_T;
extern const MPI_Datatype RSMPI_UINT16_T;
extern const MPI_Datatype RSMPI_UINT32_T;
extern const MPI_Datatype RSMPI_UINT64_T;

extern const MPI_Datatype RSMPI_DATATYPE_NULL;

extern const MPI_Comm RSMPI_COMM_WORLD;
extern const MPI_Comm RSMPI_COMM_NULL;
extern const MPI_Comm RSMPI_COMM_SELF;

extern const MPI_Group RSMPI_GROUP_EMPTY;
extern const MPI_Group RSMPI_GROUP_NULL;
extern const int RSMPI_UNDEFINED;

extern const int RSMPI_PROC_NULL;
extern const int RSMPI_ANY_SOURCE;
extern const int RSMPI_ANY_TAG;

extern const MPI_Message RSMPI_MESSAGE_NO_PROC;

extern const MPI_Request RSMPI_REQUEST_NULL;

extern const int RSMPI_IDENT;
extern const int RSMPI_CONGRUENT;
extern const int RSMPI_SIMILAR;
extern const int RSMPI_UNEQUAL;

extern const int RSMPI_THREAD_SINGLE;
extern const int RSMPI_THREAD_FUNNELED;
extern const int RSMPI_THREAD_SERIALIZED;
extern const int RSMPI_THREAD_MULTIPLE;

extern const int RSMPI_MAX_LIBRARY_VERSION_STRING;
extern const int RSMPI_MAX_PROCESSOR_NAME;
#endif
