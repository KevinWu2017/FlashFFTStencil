#include <iostream>
#include <vector>

using namespace std;

typedef vector<vector<double>> Matrix;

// 获取矩阵元素，处理循环边界
double getElement(const Matrix& mat, int x, int y) {
    int rows = mat.size();
    int cols = mat[0].size();
    // 使用取模运算处理边界
    int wrappedX = (x + rows) % rows;
    int wrappedY = (y + cols) % cols;
    return mat[wrappedX][wrappedY];
}

// 进行二维卷积操作
Matrix convolution2D(const Matrix& input, const Matrix& kernel) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = inputSize; // 输出大小与输入相同

    // 初始化输出矩阵
    Matrix output(outputSize, vector<double>(outputSize, 0.0));

    int halfKernel = kernelSize / 2;

    // 执行卷积操作
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            double sum = 0.0;
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    int x = i + m - halfKernel;
                    int y = j + n - halfKernel;
                    sum += kernel[m][n] * getElement(input, x, y);
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

// 打印矩阵
void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

int main() {
    // 输入矩阵
    Matrix input = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // 卷积核
    Matrix kernel = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };

    // 计算卷积
    Matrix output = convolution2D(input, kernel);

    // 打印输出矩阵
    cout << "Convolution Result:" << endl;
    printMatrix(output);

    return 0;
}
