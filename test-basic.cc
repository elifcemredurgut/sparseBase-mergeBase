#include <iostream>
#include <set>

#include <sparsebase/format/format.h>
#include <sparsebase/object/object.h>
#include <sparsebase/preprocess/preprocess.h>
#include <sparsebase/utils/io/reader.h>

#include <omp.h>
#include <algorithm>
#include "spmv.hpp"

#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime


//#include <mkl.h>
using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

int main(int argc, char *argv[]) {



  std::string A_filename = argv[1];
  context::CPUContext cpu_context;

  utils::io::MTXReader<unsigned long long, unsigned long long, double> A_reader(A_filename, true);
  //utils::io::MTXReader<unsigned long long, unsigned long long, double> b_reader(A_filename, true);

  format::CSR<unsigned long long, unsigned long long, double> * csr =  A_reader.ReadCSR();
  double *vector_x, *vector_y_out;
  vector_x                = (double*) calloc(sizeof(double) * csr->get_dimensions()[1], 4096);
  vector_y_out            = (double*) calloc(sizeof(double) * csr->get_dimensions()[0], 4096);
  cout << "Vector x:\n";
  for (int col = 0; col < csr->get_dimensions()[1]; ++col){
        vector_x[col] = 1.0;
        cout << vector_x[col] << " ";}

  cout << "\nMatrix values:\n";
  for(int v=0; v < csr->get_num_nnz(); ++v)
        cout << csr->get_vals()[v] << " ";

  cout << "\nRow ptr:\n";
  for(int v=0; v < 5; ++v)
        cout << csr->get_row_ptr()[v] << " ";

  //OmpMergeCsrmv<double, unsigned long long, long long unsigned int>(1, *csr, csr->get_row_ptr(), csr->get_col(), csr->get_vals(), vector_x, vector_y_out);
  CSRmv<double, unsigned long long, long long unsigned int>(*csr, csr->get_row_ptr(), csr->get_col(), csr->get_vals(), vector_x, vector_y_out);
  cout << "\nVector y:\n";
  for (int row = 0; row < csr->get_dimensions()[0]; ++row)
        cout << vector_y_out[row] << " ";
  cout << "\n";

  return 0;
}