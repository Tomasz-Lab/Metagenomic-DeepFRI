

import numpy as np

cimport cython
cimport numpy as cnp
from libcpp.vector cimport vector

ctypedef cnp.int32_t DTYPE_t
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef pairwise_sqeuclidean(float[:, ::1] X, int threads=1):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int i, j, k
    cdef float d, diff
    cdef int n_threads = threads

    cdef float[:, ::1] D = np.zeros((n, n), dtype=np.float32)

    with nogil:
        for i in prange(n, schedule='static', chunksize=1, num_threads=n_threads):
            for j in range(i + 1, n):
                d = 0.0
                for k in range(m):
                    diff = X[i, k] - X[j, k]
                    d = d + (diff * diff)

                D[i, j] = d
                D[j, i] = d

    return np.asarray(D)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] align_contact_map(str query_alignment,
                        str target_alignment,
                        cnp.ndarray[DTYPE_t, ndim=2] sparse_target_contact_map,
                        int generated_contacts=2,
                        int threads=1):
    cdef bytes query_bytes = query_alignment.encode('ascii')
    cdef bytes target_bytes = target_alignment.encode('ascii')
    cdef char* q_ptr = query_bytes
    cdef char* t_ptr = target_bytes
    cdef int align_len = len(query_bytes)
    cdef vector[int] target_to_query_map
    target_to_query_map.reserve(align_len)

    cdef vector[int] sparse_query_contacts
    sparse_query_contacts.reserve(align_len * 2)

    cdef int query_idx = 0
    cdef int target_idx = 0
    cdef int i, j, k, row

    for i in range(align_len):
        if q_ptr[i] == 45:
            target_to_query_map.push_back(-1)
            target_idx += 1
        else:
            if t_ptr[i] == 45:
                for j in range(1, generated_contacts + 1):
                    sparse_query_contacts.push_back(query_idx + j)
                    sparse_query_contacts.push_back(query_idx)
                    sparse_query_contacts.push_back(query_idx - j)
                    sparse_query_contacts.push_back(query_idx)

                query_idx += 1
            else:
                target_to_query_map.push_back(query_idx)
                query_idx += 1
                target_idx += 1

    cdef cnp.ndarray[DTYPE_t, ndim=2] output_map = np.zeros((query_idx, query_idx), dtype=np.int32)
    cdef int[:, :] output_view = output_map

    for i in range(query_idx):
        output_map[i, i] = 1

    cdef int p1, p2
    cdef size_t total_sparse_entries = sparse_query_contacts.size()

    for i in range(0, total_sparse_entries, 2):
        p1 = sparse_query_contacts[i]
        p2 = sparse_query_contacts[i+1]

        if 0 <= p1 < query_idx and 0 <= p2 < query_idx:
            output_map[p1, p2] = 1
            output_map[p2, p1] = 1

    cdef int[:, :] target_contacts_view = sparse_target_contact_map
    cdef int num_target_contacts = sparse_target_contact_map.shape[0]
    cdef int t_res_i, t_res_j
    cdef int q_res_i, q_res_j
    cdef int n_threads = threads

    with nogil:
        for row in prange(num_target_contacts, schedule='static', num_threads=n_threads):
            t_res_i = target_contacts_view[row, 0]
            t_res_j = target_contacts_view[row, 1]

            if t_res_i < target_to_query_map.size() and t_res_j < target_to_query_map.size():
                q_res_i = target_to_query_map[t_res_i]
                q_res_j = target_to_query_map[t_res_j]

                if q_res_i != -1 and q_res_j != -1:
                    output_view[q_res_i, q_res_j] = 1
                    output_view[q_res_j, q_res_i] = 1

    return output_map
