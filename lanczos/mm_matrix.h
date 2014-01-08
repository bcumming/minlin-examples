#pragma once

#include "mmio.h"

#include <string>
#include <iostream>
#include <vector>

#include <cstdio>

template <typename T>
class mm_matrix {
    std::string fname_;
    int rows_;
    int cols_;
    int nnz_;
    std::vector<T> data_;
    std::vector<size_t> row_idx_;
    std::vector<size_t> col_idx_;
    MM_typecode matcode_;

    public:

    // default, empty constructor
    mm_matrix() : rows_(0), cols_(0), nnz_(0) {};

    // default, constructor from file name
    mm_matrix(std::string file) {
        load(file);
    }

    // return whether file has been loaded
    bool is_loaded() {
        return (cols_ && rows_);
    }

    // return whether matrix is symmetric
    bool is_symmetric() {
        return (mm_is_symmetric(matcode_));
    }

    // copy into a buffer
    // target is a pointer to location where the sparse matrix should be stored
    // in dense format. At least cols_*rows_*sizeof(T) memory should have been 
    // preallocated in target before calling to_dense().
    void to_dense(T *target, bool row_major) {
        // only bother if the matrix is non-empty
        if( !is_loaded() )
            return;

        // initialize buffer to zero
        std::fill(target, target+(cols_*rows_), T(0));

        if( row_major) {
            for(size_t i=0; i<nnz_; i++)
                target[row_idx_[i]*cols_+col_idx_[i]] = data_[i];
            if( is_symmetric() )
                for(size_t i=0; i<nnz_; i++)
                    target[col_idx_[i]*cols_+row_idx_[i]] = data_[i];
        }
        else {
            for(size_t i=0; i<nnz_; i++)
                target[row_idx_[i]+col_idx_[i]*rows_] = data_[i];
            if( is_symmetric() )
                for(size_t i=0; i<nnz_; i++)
                    target[col_idx_[i]+row_idx_[i]*rows_] = data_[i];
        }
    }

    // load from file
    void load(std::string file) {
        // open file stream
        // this uses good 'ol FILE out of defernce to the matrix market routines
        FILE* fid=0;
        fid = fopen(file.c_str(), "r");

        rows_ = cols_ = nnz_ = 0;

        // check that file stream is a-ok
        if( !fid ) {
            std::cerr << "ERROR : unable to open file for input : " << file << std::endl;
            return;
        }

        // read the banner
        int status = mm_read_banner(fid, &matcode_);
        if( status ) {
            std::cerr << "ERROR : unable to open matrix market banner : " << file << std::endl;
            return;
        }

        // bail if we have a complex matrix
        if( mm_is_complex(matcode_) ) {
            std::cerr << "ERROR : this file has complex variables! : " << file << std::endl;
            return;
        }

        status = mm_read_mtx_crd_size(fid, &rows_, &cols_, &nnz_);
        if( status ) {
            std::cerr << "ERROR : unable to read matrix dims : " << file << std::endl;
            rows_ = cols_ = nnz_ = 0;
            return;
        }

        // reseve memory for matrix
        row_idx_.resize(nnz_);
        col_idx_.resize(nnz_);
        data_.resize(nnz_);

        // read in the dat points
        for (int i=0; i<nnz_; i++) {
            double val;
            int r, c;
            fscanf(fid, "%d %d %lg\n", &r, &c, &val);
            data_[i] = T(val);
            row_idx_[i] = size_t(r-1);
            col_idx_[i] = size_t(c-1);
        }

        // close the file
        fclose(fid);

        fname_ = file;
    }

    // print some stats to stdout
    void stats() {
        std::cout << fname() << " was " << (is_loaded() ? "":"not ") << "loaded " << std::endl;
        if( is_loaded() ) {
            std::cout << (is_symmetric() ? "symmetric " : "general ")
                    << "matrix has dimensions " << rows() << "*" << cols()
                    << " with " << nnz() << " non zeros : "
                    << 100.*double(nnz())/double(rows()*cols()) << "%" << std::endl;
        }
    }

    // getters
    size_t rows() {return rows_;};
    size_t cols() {return cols_;};
    size_t nnz() {return nnz_;};
    std::vector<T> &data() {return data_;};
    std::vector<size_t> &row_idx() {return row_idx_;};
    std::vector<size_t> &col_idx() {return col_idx_;};
    std::string const& fname() const {return fname_;};
};

