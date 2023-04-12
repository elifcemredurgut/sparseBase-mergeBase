#include <iostream>
#include <set>

#include <sparsebase/format/format.h>
#include <sparsebase/object/object.h>
#include <sparsebase/preprocess/preprocess.h>
#include <sparsebase/utils/io/reader.h>

#include <omp.h>
#include <algorithm>
#include "mv.hpp"

#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime


//#include <mkl.h>
using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

int main(int argc, char *argv[]) {

std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
auto duration = now.time_since_epoch();

typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<3>
>::type> Days; /* UTC: +8:00 */
Days days = std::chrono::duration_cast<Days>(duration);
    duration -= days;
auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    duration -= seconds;
auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    duration -= milliseconds;
auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    duration -= microseconds;
auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

std::cout << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << std::endl;

  std::string A_filename = argv[1];
  context::CPUContext cpu_context;

  utils::io::MTXReader<unsigned long long, unsigned long long, double> A_reader(A_filename, true);
  //utils::io::MTXReader<unsigned long long, unsigned long long, double> b_reader(A_filename, true);

  format::CSR<unsigned long long, unsigned long long, double> * csr =  A_reader.ReadCSR();
  double *vector_x, *vector_y_out;
  vector_x                = (double*) calloc(sizeof(double) * csr->get_dimensions()[1], 4096);
  vector_y_out            = (double*) calloc(sizeof(double) * csr->get_dimensions()[0], 4096);
  //cout << "Vector x:\n";
  //for (int col = 0; col < csr->get_dimensions()[1]; ++col){
    //    vector_x[col] = 1.0;
      //  cout << vector_x[col] << " ";}

  //cout << "\nMatrix values:\n";
  //for(int v=0; v < csr->get_num_nnz(); ++v)
    //    cout << csr->get_vals()[v] << " ";

  //cout << "\nRow ptr:\n";
  //for(int v=0; v < 5; ++v)
    //    cout << csr->get_row_ptr()[v] << " ";

  OmpMergeCsrmv<double, unsigned long long, long long unsigned int>(1, *csr, csr->get_row_ptr(), csr->get_col(), csr->get_vals(), vector_x, vector_y_out);
  
  //cout << "\nVector y:\n";
  //for (int row = 0; row < csr->get_dimensions()[0]; ++row)
        //cout << vector_y_out[row] << " ";

now = std::chrono::system_clock::now();
duration = now.time_since_epoch();

typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<3>
>::type> Days; /* UTC: +8:00 */
days = std::chrono::duration_cast<Days>(duration);
    duration -= days;
hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    duration -= seconds;
milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    duration -= milliseconds;
microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    duration -= microseconds;
nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

std::cout << std::endl 
          << hours.count() << ":"
          << minutes.count() << ":"
          << seconds.count() << ":"
          << milliseconds.count() << ":"
          << microseconds.count() << ":"
          << nanoseconds.count() << std::endl;
  return 0;
}