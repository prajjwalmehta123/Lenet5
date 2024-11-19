#ifndef SUBSAMPLING_H
#define SUBSAMPLING_H
#include <vector>
#include<omp.h>

using namespace std;
class subsampling {
    public:
        int output_image_size;
        subsampling();
        subsampling(int kernel_size,int stride, int image_kernel_size);
        std::vector<std::vector<float>> average_pooling(std::vector<std::vector<float>> inputBatch);
    private:
        int kernel_size;
        int stride;
        int image_kernel_size;
};



#endif //SUBSAMPLING_H
