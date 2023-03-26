all:
        g++ -o rcm order.cc -std=c++17 -fopenmp -DREORDER -DRCM -O3
        g++ -o random order.cc -std=c++17 -fopenmp -DREORDER -DRCM -DRANDOM -O3
        g++ -o degree order.cc -std=c++17 -fopenmp -DREORDER -DDEGREE -O3
        g++ -o none order.cc -std=c++17 -fopenmp -DNONE -O3