#include <cmath>
#include <vector>
#include <iostream>

bool areArraysEqual(const double *array1, const double *array2, const int length, const double epsilon)
{
    for (int i = 0; i < length; ++i)
    {
        if (std::abs(array1[i] - array2[i]) > epsilon)
        {
            return false; // 如果差异超过 epsilon，认为不相等
        }
    }
    return true; // 所有元素都在误差范围内相等
}

std::vector<std::vector<double>> rotateKernel180(const std::vector<double> &kernel)
{
    const int kernel_width = std::sqrt(kernel.size());
    std::vector<std::vector<double>> rotated_kernel(kernel_width, std::vector<double>(kernel_width, 0.0));

    for (int i = 0; i < kernel_width; ++i)
    {
        for (int j = 0; j < kernel_width; ++j)
        {
            rotated_kernel[i][j] = kernel[(kernel_width - 1 - i) * kernel_width + (kernel_width - 1 - j)];
        }
    }

    return rotated_kernel;
}

using namespace std;

vector<vector<double>> convertTo2D(double *input, const int width)
{
    vector<vector<double>> output(width, vector<double>(width));

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            output[i][j] = input[i * width + j];
        }
    }

    return output;
}

typedef vector<vector<double>> Matrix;

// 获取矩阵元素，处理循环边界
double getElement(const Matrix &mat, int x, int y)
{
    int rows = mat.size();
    int cols = mat[0].size();
    // 使用取模运算处理边界
    int wrappedX = (x + rows) % rows;
    int wrappedY = (y + cols) % cols;
    return mat[wrappedX][wrappedY];
}

// 进行二维卷积操作
vector<double> stencil2D(const Matrix &input, const Matrix &kernel)
{
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = inputSize; // 输出大小与输入相同

    // 初始化输出矩阵
    // Matrix output(outputSize, vector<double>(outputSize, 0.0));

    vector<double> output(outputSize * outputSize, 0.0);

    int halfKernel = kernelSize / 2;

    // 执行卷积操作
    #pragma omp parallel for
    for (int i = 0; i < outputSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            double sum = 0.0;
            for (int m = 0; m < kernelSize; ++m)
            {
                for (int n = 0; n < kernelSize; ++n)
                {
                    int x = i + m - halfKernel;
                    int y = j + n - halfKernel;
                    sum += kernel[m][n] * getElement(input, x, y);
                }
            }

            int index_i = (i + kernelSize / 2) % outputSize;
            int index_j = (j + kernelSize / 2) % outputSize;

            output[index_i * outputSize + index_j] = sum;
        }
    }
    return output;
}