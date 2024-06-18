#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <fstream>
#include <sstream>
#include <cfloat>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <stdatomic.h>
#include <cmath>
#include <parallel/algorithm>  
#include <unordered_map>
#include <random>
#include <atomic>
#include <string>
#include <omp.h>
#include <iostream>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;
int pre = 0;
int direction_to_index_mapping[12][3] = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};   

// g++-12 -fopenmp -std=c++17 -O3 -g preserve3d_omp_d.cpp -o helloworldomp_d
// 4.from_direction_to_index1
// g++ -fopenmp hello2.o kernel.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld

int maxNeighbors = 12;
int width;
int height;
int depth;
int size2;
int un_sign_as;
int un_sign_ds;
std::vector<int> adjacency;
int ite = 0;
std::vector<double> d_deltaBuffer;
// std::vector<double> d_deltaBuffer;
std::vector<int> all_max, all_min, all_d_max, all_d_min;
atomic_int count_max = 0;
atomic_int count_min = 0;
atomic_int count_f_max = 0;
atomic_int count_f_min = 0;

std::vector<int> record;
std::vector<std::vector<float>> record1;
std::vector<std::vector<float>> record_ratio;
// 没有0，0 和 1，1
int directions1[36] =  {0,1,0,0,-1,0,1,0,0,-1,0,0,-1,1,0,1,-1,0,0,0, -1,  0,-1, 1, 0,0, 1,  0,1, -1,  -1,0, 1,   1, 0,-1};

// std::vector<double> getdata(std::string filename){
    
//      std::vector<double> data;
//      data.resize(width*height*depth);
//      std::string line;
//      std::ifstream file(filename);
//     //  std::ifstream file(filename);
//     // std::cout<<"dh"<<std::endl;
//     // std::vector<double> data;
//     double num;

//     if (!file.is_open()) {
//         std::cerr << "Error opening file" << std::endl;
//         return data;
//     }
//     int cnt=0;
//     while (!file.eof()) { // 循环直到文件结束
//         if (file >> num) {
//             data[cnt] = num;
//             cnt+=1;
//         } 
//         else {
            
//             file.clear(); // 清除失败的状态
//             std::string invalid_input;
//             if (file >> invalid_input) {
//                 std::cout << "无效的输入: " << invalid_input << std::endl;
//             }
//             // file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略当前行的剩余部分
//             if (file.eof()) break;
//         }
//     }
//     // std::cout<<"读完了"<<cnt<<std::endl;
//     file.close();
    
//     // std::cout<<data.size()<<std::endl;
//     // Optionally print the data to verify
//     // for (double num : data) {
//     //     std::cout << num << " ";
//     // }
//     // std::cout << std::endl;

//      // 读取每一行
//     //  while (std::getline(file, line)) {
//     //       std::vector<double> row;
//     //       std::stringstream ss(line);
//     //       std::string value;

//     //       // 分割行中的每个值
//     //       while (std::getline(ss, value, ',')) {
//     //         try {
//     //             double number = std::stod(value);
//     //             // 后续操作
//     //         } catch (const std::exception& e) {
//     //             if(typeid(value) == typeid(std::string)){
//     //                 std::cout<<value<<std::endl;
//     //             }
                
//     //             std::cerr << "转换错误: " << e.what() << '\n';
//     //             // 异常处理代码
//     //         }

//     //         // std::cout<<value<<std::endl;
//     //            data.push_back(std::stod(value)); // 将字符串转换为 double
//     //       }

//     //     //   data.push_back(row);
//     //  }

//     //  file.close();
     
//      return data;
// };
// std::vector<double> getdata1(std::string filename){
//     std::vector<double> myVector;

//     // 打开一个文件流
//     std::ifstream inFile("output.txt");

//     // 检查文件是否成功打开
//     if (inFile.is_open()) {
//         double value;
//         // 读取文件中的每个整数，并添加到 vector 中
//         while (!inFile.eof()) { // 循环直到文件结束
//             if (inFile >> value) {
//                 myVector.push_back(value);
                
//             } 
//             else {
                
//                 inFile.clear(); // 清除失败的状态
//                 std::string invalid_input;
//                 if (inFile >> invalid_input) {
//                     std::cout << "无效的输入: " << invalid_input << std::endl;
//                 }
//                 // file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略当前行的剩余部分
//                 if (inFile.eof()) break;
//             }
//         }
        
        
//         // 关闭文件流
//         inFile.close();
//     } 
    
    
    
//     return myVector;
// }
// std::string inputfilename= "result_3d.txt";

std::vector<double> getdata2(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::vector<double> data;
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return data;
    }

    // 获取文件大小
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 确定文件中有多少个float值
    std::streamsize num_floats = size / sizeof(double);
    // std::cout<<num_floats<<std::endl;
    // 创建一个足够大的vector来存储文件内容
    std::vector<double> buffer(num_floats);

    // 读取文件内容到vector
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        // std::cout << "文件读取成功" << std::endl;

        // 输出读取的数据，以验证
        // for (double num : buffer) {
        //     std::cout << num << " ";
        // }
        // std::cout << std::endl;
        return buffer;
    } else {
        std::cerr << "文件读取失败" << std::endl;
        return buffer;
    }
    
    return buffer;
}
std::string inputfilename;
// std::string decompfilename= "result_3d_decomp.txt";
// std::string decompfilename= "output.txt";
std::string decompfilename;

// cnpy::NpyArray arr = cnpy::npy_load("result3d.npy");
// float* loaded_data = arr.data<float>();






// std::vector<int> ordered_input_data = processdata(input_data);

std::vector<double> decp_data;
std::vector<double> input_data;
// std::vector<double> bound_data;
std::unordered_map<int, double> maxrecord;
std::unordered_map<int, double> minrecord;
double bound;
// std::vector<int> findUniqueElements(const std::unordered_set<int>& set1, const std::unordered_set<int>& set2) {
//     std::vector<int> uniqueElements;
//     // 检查 set1 中有而 set2 中没有的元素
//     for (const auto& elem : set1) {
//         if (set2.find(elem) == set2.end()) {
//             uniqueElements.push_back(elem);
//         }
//     }

//     // 检查 set2 中有而 set1 中没有的元素
//     for (const auto& elem : set2) {
//         if (set1.find(elem) == set1.end()) 
//         {
//             uniqueElements.push_back(elem);
//         }
//     }

//     return uniqueElements;
// }


std::vector<int> find_low(){
    std::vector<int> lowGradientIndices(size2, 0);
    
    const double threshold =  1e-16; // 梯度阈值
    // 遍历三维数据计算梯度
    for (int i = 0; i < width; ++i) {
        
        for (int j = 0; j < height; ++j) {
            
            for (int k = 0; k < depth; ++k) {
                
                int rm = i  + j * width + k * (height * width);
                
                
                // for(int q=0;q<12;q++){
                for(int q=0;q<12;q++){
                // for (auto& dir : directions1) {

                    int newX = i + directions1[q*3];
                    int newY = j + directions1[q*3+1];
                    int newZ = k + directions1[q*3+2];
                    int r = newX  + newY * width + newZ* (height * width);
                    if(r>=0 and r<size2){
                        double gradZ3 = abs(input_data[r] - input_data[rm])/2;
                        if (gradZ3<=threshold) {
                            lowGradientIndices[rm]=1;
                            lowGradientIndices[r]=1;
                        }
                    // }
                }
                }
            }
        }
    }
    // std::cout<<"ok"<<std::endl;
    return lowGradientIndices;
}

std::vector<int> lowGradientIndices;
void computeAdjacency() {
    // #pragma omp parallel for
    for(int i=0;i<size2;i++){
        if (lowGradientIndices[i]==0) {

        int y = (i / (width)) % height; // Get the x coordinate
        int x = i % width; // Get the y coordinate
        int z = (i / (width * height)) % depth;
        int neighborIdx = 0;
        
        for (int d = 0; d < 12; d++) {
            
            int dirX = directions1[d * 3];     
            int dirY = directions1[d * 3 + 1]; 
            int dirZ = directions1[d * 3 + 2]; 
            int newX = x + dirX;
            int newY = y + dirY;
            int newZ = z + dirZ;
            int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
            // if(lowGradientIndices[r]==1){
            //     continue;
            // }
            if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0 && lowGradientIndices[r]==0) {
                
                adjacency[i * maxNeighbors + neighborIdx] = r;
                neighborIdx++;

            }
        }

        // Fill the remaining slots with -1 or another placeholder value
        
        for (int j = neighborIdx; j < maxNeighbors; ++j) {
            adjacency[i * maxNeighbors + j] = -1;
        }
    }
    
    }
}

std::vector<std::vector<int>> _compute_adjacency(){
    std::vector<std::vector<int>> adjacency;
    for (int i = 0; i < size2; ++i) {
            int y = (i / (height)) % width; // Get the x coordinate
            int x = i % height; // Get the y coordinate
            int z = i / (width * height);
            std::vector<int> adjacency_temp;
            for (int q=0;q<12;q++) {

                int newX = x + directions1[q*3];
                int newY = y + directions1[q*3+1];
                int newZ = z + directions1[q*3+2];
                int r = newX + newY  * height + newZ* (height * width); // Calculate the index of the adjacent vertex
                
                
                // Check if the new coordinates are within the bounds of the mesh
                if (newX >= 0 && newX < height && newY >= 0 && newY < width && r < width*height*depth && newZ<depth && newZ>=0 && lowGradientIndices[r] != 1) {
                    
                    adjacency_temp.push_back(r);
                }
                // if(input_data[r]-input_data[i]==0 and input_data[r]==0){
                //     continue;
                // }
            }
            adjacency.push_back(adjacency_temp);
        }
    return adjacency;
}


// std::vector<double> add_noise(const std::vector<double>& data, double x) {
//     std::vector<double> noisy_data = data;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis(-x, x);

//     std::for_each(noisy_data.begin(), noisy_data.end(), [&dis, &gen](double &d){
//         d += dis(gen);
//     });

//     return noisy_data;
// };

std::map<std::tuple<int, int, int>, int> createDirectionMapping() {
    std::map<std::tuple<int, int, int>, int> direction_mapping_3d;
    direction_mapping_3d[std::make_tuple(0,1,0)] = 1;
    direction_mapping_3d[std::make_tuple(0,-1,0)] = 2;
    direction_mapping_3d[std::make_tuple(1,0,0)] = 3;
    direction_mapping_3d[std::make_tuple(-1,0,0)] = 4;
    direction_mapping_3d[std::make_tuple(-1,1,0)] = 5;
    direction_mapping_3d[std::make_tuple(1,-1,0)] = 6;

    // Additional 3D directions
    direction_mapping_3d[std::make_tuple(0, 0, -1)] = 7;   // down in Z
    direction_mapping_3d[std::make_tuple(0,-1, 1)] = 8;   // down-left in Z
    direction_mapping_3d[std::make_tuple(0, 0, 1)] = 9;    // up in Z
    direction_mapping_3d[std::make_tuple(0, 1, -1)] = 10;  // up-right in Z
    direction_mapping_3d[std::make_tuple(-1, 0, 1)] = 11;  // left-up in Z
    direction_mapping_3d[std::make_tuple(1, 0, -1)] = 12; 

    return direction_mapping_3d;
};

int a = 0;
// std::vector<int> processdata(std::vector<double> data){
//     std::vector<std::pair<double, int>> value_index_pairs;
    
//     int n = data.size();
    
//     // value_index_pairs.clear();
    
//     for (int i = 0; i < n; ++i) {
//         value_index_pairs.emplace_back(data[i], i);
//     }

//     __gnu_parallel::stable_sort(value_index_pairs.begin(), value_index_pairs.end(),[](const std::pair<double, int>& a, const std::pair<double, int>& b) {
//             if (a.first == b.first) {
//                 return a.second < b.second; // 注意这里是‘>’
//             }
//             return a.first < b.first;
//     });
    
    
//     std::vector<int> sorted_indices(n);
//     // #pragma omp parallel for
//     for (int i = 0; i < n; ++i) {
        
//         sorted_indices[value_index_pairs[i].second] = i;
//     }
    
//     return sorted_indices;

// };
// std::string inputfilename= "input_arrayc"+std::to_string(width)+".csv";


// std::vector<double> decp_data = add_noise(input_data,bound);

std::map<int, std::tuple<int, int, int>> createReverseMapping(const std::map<std::tuple<int, int ,int>, int>& originalMap) {
    std::map<int, std::tuple<int, int, int>> reverseMap;
    for (const auto& pair : originalMap) {
        reverseMap[pair.second] = pair.first;
    }
    return reverseMap;
}
std::map<std::tuple<int, int ,int>, int> direction_mapping = createDirectionMapping();

std::map<int, std::tuple<int, int ,int>> reverse_direction_mapping = createReverseMapping(direction_mapping);
class myTriangularMesh {
    public:
        std::vector<double> values;
        // std::vector<std::vector<int>> adjacency;
        std::unordered_set<int> maxi;
        std::unordered_set<int> mini;
        // 函数声明

        // void _compute_adjacency();
        // void get_criticle_points();
        // std::vector<int> adjacent_vertices( int index);

}; 

// void myTriangularMesh::get_criticle_points(){
    
//     std::vector<int> global_maxi_temp;
//     std::vector<int> global_mini_temp;

//     #pragma omp parallel
//     {
//         std::vector<int> local_maxi_temp;
//         std::vector<int> local_mini_temp;

//         #pragma omp for nowait
//         for (int i = 0; i < size2; ++i) {
//             if(lowGradientIndices[i] == 1){
//                 continue;
//             }
//             bool is_maxima = true;
//             bool is_minima = true;

//             for (int j : adjacency[i]) {
//                 if(lowGradientIndices[j] == 1){
//                     continue;
//                 }
//                 if (this->values[j] > this->values[i]) {
//                     is_maxima = false;
//                     break;
//                 }
//                 else if(this->values[j] == this->values[i] and j>i){
//                     is_maxima = false;
//                     break;
//                 }
//             }
//             for (int j : adjacency[i]) {
//                 if(lowGradientIndices[j] == 1){
//                     continue;
//                 }
//                 if (this->values[j] < this->values[i]) {
//                     is_minima = false;
//                     break;
//                 }
//                 else if(this->values[j] == this->values[i] and j<i){
//                     is_minima = false;
//                     break;
//                 }
//             }

//             if (is_maxima and lowGradientIndices[i] == 0) {
//                 local_maxi_temp.push_back(i);
//             }
//             if (is_minima and lowGradientIndices[i] == 0) {
//                 local_mini_temp.push_back(i);
//             }
//         }

//         #pragma omp critical
//         {
//             global_maxi_temp.insert(global_maxi_temp.end(), local_maxi_temp.begin(), local_maxi_temp.end());
//             global_mini_temp.insert(global_mini_temp.end(), local_mini_temp.begin(), local_mini_temp.end());
//         }
//     }

//     // std::unordered_set<int> mesh_maxi(global_maxi_temp.begin(), global_maxi_temp.end());
//     // std::unordered_set<int> mesh_mini(global_mini_temp.begin(), global_mini_temp.end());
//     // #pragma omp parallel for
//     //     for (int i = 0; i < size2; ++i) {
//     //         if (maximum[i] == -1) {
//     //             // #pragma omp critical
//     //             maxi_temp.push_back(i);
//     //         }
//     //     }
//     // // #pragma omp parallel for
//     //     for (int i = 0; i < size2; ++i) {
//     //         if (minimum[i] == -1) {
//     //             // #pragma omp critical
//     //             mini_temp.push_back(i);
//     //         }
//     //     }
//     std::unordered_set<int> mesh_maxi(global_maxi_temp.begin(), global_maxi_temp.end());
//     std::unordered_set<int> mesh_mini(global_mini_temp.begin(), global_mini_temp.end());
//     this->maxi = mesh_maxi;
//     this->mini = mesh_mini;
    

// };

// void myTriangularMesh::get_criticle_points(){
//     std::vector<int> maxi_temp;
//     std::vector<int> mini_temp;
//     std::vector<int> maximum(size2, 0);
//     std::vector<int> minimum(size2, 0);
    
   
//     for (int i = 0; i < size2; ++i) {
//             bool is_maxima = true;
//             bool is_minima = true;

            
//             for (int j : adjacency[i]) {
//                 if (this->values[j] > this->values[i]) {
//                     is_maxima = false;
//                     break;
//                 }
//                 if(this->values[j] == this->values[i] and j>i){
//                     is_maxima = false;
//                     break;
//                 }
//             }
//             for (int j : adjacency[i]) {
//                 if (this->values[j] < this->values[i]) {
//                     is_minima = false;
//                     break;
//                 }
//                 if(this->values[j] == this->values[i] and j<i){
//                     is_minima = false;
//                     break;
//                 }
//             }
    

            

//             if (is_maxima) {
//                 maximum[i]=-1; // Add index to maxima list
//             }
//             if (is_minima) {
//                 minimum[i]=-1; // Add index to minima list
//             }
//         }
    
    
    

//     // #pragma omp parallel for
//         for (int i = 0; i < size2; ++i) {
//             if (maximum[i] == -1) {
//                 // #pragma omp critical
//                 maxi_temp.push_back(i);
//             }
//         }
//     // #pragma omp parallel for
//         for (int i = 0; i < size2; ++i) {
//             if (minimum[i] == -1) {
//                 // #pragma omp critical
//                 mini_temp.push_back(i);
//             }
//         }
//     std::unordered_set<int> mesh_maxi(maxi_temp.begin(), maxi_temp.end());
//     std::unordered_set<int> mesh_mini(mini_temp.begin(), mini_temp.end());
//     this->maxi = mesh_maxi;
//     this->mini = mesh_mini;
// };



// std::vector<double> linspace(double start, double end, int num) {
//     std::vector<double> linspaced;

//     if (num == 0) { 
//         return linspaced; 
//     }
//     if (num == 1) {
//         linspaced.push_back(start);
//         return linspaced;
//     }

//     double delta = (end - start) / (num - 1);

//     for(int i=0; i < num-1; ++i) {
//         linspaced.push_back(start + delta * i);
//     }
//     linspaced.push_back(end); 
    
//     // 确保end被包括进去
//     return linspaced;
// }


// std::vector<int> ordered_decp_data = processdata(decp_data);

myTriangularMesh or_mesh;
myTriangularMesh de_mesh;
std::vector<int> wrong_maxi_cp;
std::vector<int> wrong_min_cp;
std::vector<int> wrong_index_as;
std::vector<int> wrong_index_ds;

int getDirection(int x, int y, int z){
    
    for (int i = 0; i < 12; ++i) {
        if (direction_to_index_mapping[i][0] == x && direction_to_index_mapping[i][1] == y && direction_to_index_mapping[i][2] == z) {
            return i+1;  
        }
    }
    return -1;  

// 26302898,3378820
// 27930227,32438238
}

int from_direction_to_index1(int cur, int direc){
    
    if (direc==-1) return cur;
    int x = cur % width;
    int y = (cur / width) % height;
    int z = (cur/(width * height))%depth;
    // printf("%d %d\n", row, rank1);
    if (direc >= 1 && direc <= 12) {
        int delta_row = direction_to_index_mapping[direc-1][0];
        int delta_col = direction_to_index_mapping[direc-1][1];
        int delta_dep = direction_to_index_mapping[direc-1][2];
        
        
        int next_row = x + delta_row;
        int next_col = y + delta_col;
        int next_dep = z + delta_dep;
        // printf("%d \n", next_row * width + next_col);
        // return next_row * width + next_col + next_dep* (height * width);
        return next_row + next_col * width + next_dep* (height * width);
    }
    else {
        return -1;
    }
    // return 0;
};
// int find_direction (int index,  std::vector<double> &data ,int direction){
    
   
//     double mini = 0;
//     int size1 = adjacency[index].size();
    
//     std::vector<int> indexs = adjacency[index];
//     int largetst_index = index;
    
//     if (direction == 0){
        
//         for(int i =0;i<size1;++i){
//             if(lowGradientIndices[indexs[i]] == 1){
//                 continue;
//             }
//             if((data[indexs[i]]>data[largetst_index] or (data[indexs[i]]==data[largetst_index] and indexs[i]>largetst_index)) and lowGradientIndices[indexs[i]] == 0){
//                 mini = data[indexs[i]]-data[index];
                
//                 largetst_index = indexs[i];
//                 // }
                
//             };
//         };
        
        
//     }
//     else{
//         for(int i =0;i<size1;++i){
//             if(lowGradientIndices[indexs[i]] == 1){
//                 continue;
//             }
//             if((data[indexs[i]]<data[largetst_index] or (data[indexs[i]]==data[largetst_index] and indexs[i]<largetst_index)) and lowGradientIndices[indexs[i]] == 0){
                
//                 mini = data[indexs[i]]-data[index];
                
//                 largetst_index = indexs[i];
//                 // 113618, 113667, 111168, 111168
//                 // if(wrong_index_ds.size()==1 and index==111168){
//                 //     std::cout<<index<<", "<<largetst_index<<","<<decp_data[largetst_index]<<", "<<decp_data[113667]<<std::endl;
//                 // }

                
//             };
//         };
//     };
    
//     // if(lowGradientIndices[largest_index] == 1){
//     //     std::cout<<largetst_index<<std::endl;
//     // }
//     // int row_l = (largetst_index % (height * width)) / width;
//     // int row_i = (index % (height * width)) / width;
    
//     // int row_diff = row_l - row_i;
//     // int col_diff = (largetst_index % width) - (index % width);

//     // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
//     int row_l = (largetst_index / (height)) % width;
//     int row_i = (index / (height)) % width;
    
//     int col_diff = row_l - row_i;
//     int row_diff = (largetst_index % height) - (index % height);

//     int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
//     int d = getDirection(direction_mapping, row_diff, col_diff,dep_diff);
    
    
//     return d;

// };
std::vector<int> or_direction_as;
std::vector<int> or_direction_ds;
std::vector<int> de_direction_as;
std::vector<int> de_direction_ds;
int find_direction (std::vector<double> data, std::vector<int>& direction_as, std::vector<int>& direction_ds, int type=0){
        
        #pragma omp parallel for
        
        for (int index=0;index<size2;index++){
            if(lowGradientIndices[index] == 1){
                continue;
            }
                
            
            double mini = 0;
        
        
            int largetst_index = index;
    
    
        
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            continue;
        }
        if(lowGradientIndices[i]==1){
            continue;
        }
        if((data[i]>data[largetst_index] or (data[i]==data[largetst_index] and i>largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;
            // }
            
        };
    };
    // int row_l = (largetst_index % (height * width)) / width;
    // int row_i = (index % (height * width)) / width;
    
    // int row_diff = row_l - row_i;
    // int col_diff = (largetst_index % width) - (index % width);

    // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    // int x_l = (largetst_index / (height)) % width;
    // int x_i = (index / (height)) % width;
    int y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
    // int y_diff = row_l - row_i;
    int x_diff = (largetst_index % width) - (index % width);

    int z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
    direction_as[index] = getDirection(x_diff, y_diff,z_diff);
    
    // if(index==24654784 and type==0){
        
    //     printf("值：");
    //     printf("%d %d %d\n",row_diff, col_diff,dep_diff);
    //     printf("%d %d \n", largetst_index % 750, index % 750);
    //     // printf("%f %f \n" ,decp_data[index],input_data[index]);
    //     // for(int i=0;i<12;i++){
    //     //     int j = adjacency[index*12+i];
    //     //     if(j==-1){
    //     //         break;
    //     //     }
    //     //     printf("%f %f \n" ,decp_data[j],input_data[j]);
    //     // }
        
    // }
    
    

    mini = 0;
    largetst_index = index;
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowGradientIndices[i]==1){
            continue;
        }
        // if(i==8186 and index==8058 and type==0){
        //     printf("%.20f %.20f\n",data[i]-data[index],data[8057]-data[index]);
        //     // std::cout<<data[i]<<", "<<data[index]<<", "<<data[8057]<<std::endl;
        // }
        if((data[i]<data[largetst_index] or (data[i]==data[largetst_index] and i<largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;

            
        };
    };
    
    
    // row_l = (largetst_index % (height * width)) / width;
    // row_i = (index % (height * width)) / width;
    
    // row_diff = row_l - row_i;
    // col_diff = (largetst_index % width) - (index % width);

    // dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    y_diff = (largetst_index / (height)) % width - (index / (height)) % width;
    // int y_diff = row_l - row_i;
    x_diff = (largetst_index % height) - (index % height);

    z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
    // direction_as[index] = getDirection(x_diff, y_diff,z_diff);
    direction_ds[index] = getDirection(x_diff, y_diff,z_diff);
    
    
        
    };
        
    
    return 0;

};
// int find_direction1 (int index,std::map<int,int> &data ,int direction){
void applyDeltaBuffer() {
    for(int i=0;i<size2;i++){
        if(lowGradientIndices[i]!=1 and d_deltaBuffer[i]!=-2000){
            decp_data[i] += d_deltaBuffer[i];
        }
        
    }
    
}
void initialization() {
    // atomic_int init_value = -1;
   for(int i =0;i<size2;i++){
    // std::cout<<i<<std::endl;
        d_deltaBuffer[i] = -2000.0;
   }
}


// bool atomicCAS(double* ptr, double old_val, double new_val) {
//     // Use OpenMP critical section as a substitute for atomic CAS for double
//     bool swapped = false;
//     #pragma omp critical
//     {
//         if (*ptr == old_val) {
//             *ptr = new_val;
//             swapped = true;
//         }
//     }
//     return swapped;
// }

bool atomicCASDouble(double* ptr, double old_val, double new_val) {
    // 将 double 指针转换为 uint64_t 指针
    bool swapped = false;
    #pragma omp critical
    {
        if (*ptr == old_val) {
            *ptr = new_val;
            swapped = true;
        }
    }
    return swapped;
}

int swap(int index, double delta){
    int update_successful = 0;
    double oldValue = d_deltaBuffer[index];
    double new_edit = delta;
    double old_edit = d_deltaBuffer[index];
    while (new_edit > old_edit) {
        if (atomicCASDouble(&d_deltaBuffer[index],old_edit, new_edit)) {
            break;
        }
        old_edit = d_deltaBuffer[index];
    }
    
    // while (update_successful==0) {
    //     double current_value = d_deltaBuffer[index];
    //     if (delta > current_value) {
    //         double swapped = atomicCASDouble(&d_deltaBuffer[index],d_deltaBuffer[index], delta);
    //         if (swapped == current_value) {
    //             update_successful = 1;
                 
    //         } else {
    //             oldValue = swapped;
    //         }
    //     } else {
    //         update_successful = 1; 
    // }
    // }

    return 0;
}
//     double mini = 0;
//     int size1 = adjacency[index].size();
//     std::vector<int> indexs = adjacency[index];
//     int largetst_index;
    
//     if (direction == 0){
        
//         for(int i =0;i<size1;++i){
            
//             if(data[indexs[i]]-data[index]>mini){
//                 mini = data[indexs[i]]-data[index];
//                 largetst_index = indexs[i];
//             };
//         };
        
        
//     }
//     else{
//         for(int i =0;i<size1;++i){
//             if(data[adjacency[index][i]]-data[index]<mini){
//                 mini = data[adjacency[index][i]]-data[index];
//                 largetst_index = adjacency[index][i];
//             };
//         };
//     };
//     // std::cout<<largetst_index<<","<<index<<std::endl;
//     // int row_l = (largetst_index % (height * width)) / width;
//     // int row_i = (index % (height * width)) / width;
    
//     // int row_diff = row_l - row_i;
//     // int col_diff = (largetst_index % width) - (index % width);

//     // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
//     int row_l = (largetst_index / (height)) % width;
//     int row_i = (index / (height)) % width;
    
//     int col_diff = row_l - row_i;
//     int row_diff = (largetst_index % height) - (index % height);

//     int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
//     int d = getDirection(direction_mapping, row_diff, col_diff,dep_diff);
    
    
//     return d;

// };
// std::vector<int> mappath(myTriangularMesh& mesh, std::vector<int> direction_as, std::vector<int> direction_ds){
//     // std::vector<std::pair<int, int>> label(size2, std::make_pair(-1, -1));
//     std::vector<int> label(size2*2, -1);

//     #pragma omp parallel for
//     for (int i = 0;i<size2;++i){
//         if(lowGradientIndices[i] == 1){
//             continue;
//         }
        
//         int cur = i;
        
//         while (direction_as[cur]!=-1){
//             // if(lowGradientIndices.find(cur)!=lowGradientIndices.end()){
//             //     std::cout<<cur<<", "<<i<<std::endl;
//             // }
//             int direc = direction_as[cur];
            
//             int rank1 = (cur / (height)) % width;
//             int row1 = cur%height;
//             int depth1 = cur/(width * height);
            
//             int next_vertex = from_direction_to_index1(cur, direc);
            
            
//             cur = next_vertex;
            
//             if (label[cur*2+1] != -1){
//                 cur = label[cur*2+1];
//                 break;
//             }
            
//         }

//         if(direction_as[i]!=-1){
//             label[i*2+1] = cur;
//         }
//         else{
//             label[i*2+1] = -1;
//         };
        
//         cur = i;

//         while (direction_ds[cur]!=-1){
            
//             int direc = direction_ds[cur];
//             int rank1 = (cur / (height)) % width;
//             int row1 = cur%height;
//             int depth1 = cur/(width * height);
            
//             int next_vertex = from_direction_to_index1(cur, direc);
            

//             cur = next_vertex;

//             if (label[cur*2+0] != -1){
//                 cur = label[cur*2+0];
//                 break;
//             }
//         }

//         if(direction_ds[i]!=-1){
//             label[i*2+0] = cur;
//         }
//         else{
//             label[i*2+0] = -1;
//         };
        
    
//     };
    
//     return label;
// };
void initializeWithIndex(std::vector<int>& label, std::vector<int> direction_ds,std::vector<int> direction_as) {
    for(int index=0;index<size2;index++){
        
        if(direction_ds[index]!=-1){
            label[index*2] = index;
        }
        else{
            label[index*2] = -1;
        }

        if(direction_as[index]!=-1){
            label[index*2+1] = index;
        }
        else{
            label[index*2+1] = -1;
        }
    }
}
void getlabel1(std::vector<int>& label, int& un_sign_ds, int& un_sign_as, int type=0){
    std::vector<int> direction_as;
    std::vector<int> direction_ds;
    un_sign_ds = 0;
    un_sign_as = 0;
    if(type==0){

        direction_as = de_direction_as;
        direction_ds = de_direction_ds;

    }
    else{

        direction_as = or_direction_as;
        direction_ds = or_direction_ds;

    }

    #pragma omp parallel for reduction(+:un_sign_as) reduction(+:un_sign_ds)

    for(int i=0;i<size2;i++){
        
        if(lowGradientIndices[i]==1){
            continue;
        }

    
        int cur = label[i*2+1];
    
    
        int next_vertex;
        // cur!=-1就说明它首先不是cp，direction_as[cur]也说明他不是cp
        if (cur!=-1 and direction_as[cur]!=-1){
            
            int direc = direction_as[cur];
            // 找到他的下一个邻居
            
            next_vertex = from_direction_to_index1(cur, direc);
            
            // 检查下一个邻居是否为cp，如果是，直接把label换成邻居
            // if(label[next_vertex*2+1] == -1){
            label[i*2+1] = next_vertex;
                
            // }
            
            // else{
                
            //     label[i*2+1] = label[next_vertex*2+1];
                
                
            // }
            
            if (direction_as[label[i*2+1]] != -1){
                un_sign_as+=1;  
            }
            
        }
    
    
    
    
        cur = label[i*2];
        int next_vertex1;
        
        
        if (cur!=-1 and label[cur*2]!=-1){
            
            int direc = direction_ds[cur];
            // 找到他的下一个邻居
            next_vertex1 = from_direction_to_index1(cur, direc);
            // 检查下一个邻居是否为cp，如果是，直接把label换成邻居
            // if(label[next_vertex1*2] == -1){
            label[i*2] = next_vertex1;
                
            // }
            // 如果不是cp，检查邻居是否找到cp，如果找到了，就换成邻居的label
            // else if(label[label[next_vertex1*2]*2] == -1){
            //     label[i*2] = label[next_vertex1*2];  
            // }
            
            // else if(direction_ds[i]!=-1){
            //     // 如果邻居不是cp，那就替换成邻居的当前邻居
            //     if(label[next_vertex1*2]!=-1){
            //         label[i*2] = label[next_vertex1*2];
            //     }
            //     // 否则：下一个邻居是cp, 那么他的cp就是下一个邻居
            //     else{

            //         label[i*2] = next_vertex1;
            //     }
                
                
            // }
            // if(i==66590){
            //     printf("%d %d %d %d %d\n",next_vertex,de_direction_as[next_vertex],de_direction_as[label[next_vertex*2+1]],label[next_vertex*2+1],label[i*2+1]);
            // }
            if (direction_ds[label[i*2]]!=-1){
                un_sign_ds+=1;
                }
            } 
    }
    
        
    return;

}
void getlabel(std::vector<int>& label, int& un_sign_ds, int& un_sign_as, int type=0){

    std::vector<int> direction_as;
    std::vector<int> direction_ds;
    un_sign_ds = 0;
    un_sign_as = 0;
    if(type==0){

        direction_as = de_direction_as;
        direction_ds = de_direction_ds;

    }
    else{

        direction_as = or_direction_as;
        direction_ds = or_direction_ds;

    }
    
    
    #pragma omp parallel for reduction(+:un_sign_as) reduction(+:un_sign_ds)

    for(int i=0;i<size2;i++){
        
        if(lowGradientIndices[i]==1){
            continue;
        }

    
        int cur = label[i*2+1];
    
    
        int next_vertex;
        // cur!=-1就说明它首先不是cp，direction_as[cur]也说明他不是cp
        if (cur!=-1 and direction_as[cur]!=-1){
            
            int direc = direction_as[cur];
            // 找到他的下一个邻居
            
            next_vertex = from_direction_to_index1(cur, direc);
            
            // 检查下一个邻居是否为cp，如果是，直接把label换成邻居
            if(label[next_vertex*2+1] == -1){
                label[i*2+1] = next_vertex;
                
            }
            
            else{
                
                label[i*2+1] = label[next_vertex*2+1];
                
                
            }
            
            if (direction_as[label[i*2+1]] != -1){
                un_sign_as+=1;  
            }
            
        }
    
    
    
    
        cur = label[i*2];
        int next_vertex1;
        
        
        if (cur!=-1 and label[cur*2]!=-1){
            
            int direc = direction_ds[cur];
            // 找到他的下一个邻居
            next_vertex1 = from_direction_to_index1(cur, direc);
            // 检查下一个邻居是否为cp，如果是，直接把label换成邻居
            if(label[next_vertex1*2] == -1){
                label[i*2] = next_vertex1;
                
            }
            // 如果不是cp，检查邻居是否找到cp，如果找到了，就换成邻居的label
            else if(label[label[next_vertex1*2]*2] == -1){
                label[i*2] = label[next_vertex1*2];  
            }
            
            else if(direction_ds[i]!=-1){
                // 如果邻居不是cp，那就替换成邻居的当前邻居
                if(label[next_vertex1*2]!=-1){
                    label[i*2] = label[next_vertex1*2];
                }
                // 否则：下一个邻居是cp, 那么他的cp就是下一个邻居
                else{

                    label[i*2] = next_vertex1;
                }
                
                
            }
            // if(i==66590){
            //     printf("%d %d %d %d %d\n",next_vertex,de_direction_as[next_vertex],de_direction_as[label[next_vertex*2+1]],label[next_vertex*2+1],label[i*2+1]);
            // }
            if (direction_ds[label[i*2]]!=-1){
                un_sign_ds+=1;
                }
            } 
    }
    
        
    return;

    

}

void mappath1(std::vector<int>& label, int type=0){

    
    int h_un_sign_as = size2;
    int h_un_sign_ds = size2;
    
    
    if(type==0){
        initializeWithIndex(label,de_direction_ds,de_direction_as);
    }
    else{
        initializeWithIndex(label,or_direction_ds,or_direction_as);
    }
    
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        h_un_sign_as=0;
        h_un_sign_ds=0;
        
        getlabel(label,h_un_sign_as,h_un_sign_ds,type);
        
    }   

    return;
};

void mappath2(std::vector<int>& label, std::vector<double> data,int type=0){
    // data = input_data;
    initializeWithIndex(label,or_direction_ds,or_direction_as);
    double mini;
    for(int i =0;i<size2;i++){
        // std::cout<<i<<std::endl;
        if(lowGradientIndices[i]==1) continue;
        if(label[i*2+1]!=-1){
        
        int cur = label[i*2+1];
        while(cur != -1 and label[cur*2+1]!=-1)
        {
            // std::cout<<cur<<label[cur*2+1]<<or_direction_as[cur*2+1]<<std::endl;
            int largetst_index = cur;
            for(int j =0;j<12;++j){
                int q = adjacency[cur*12+j];
                
                if(q==-1){
                    continue;
                }
                if(lowGradientIndices[q]==1){
                    continue;
                }
                if((data[q]>data[largetst_index] or (data[q]==data[largetst_index] and q>largetst_index))){
                    
                    
                    largetst_index = q;
                    // }
                    
                };
            };
            if(label[largetst_index*2+1]!=-1){
                label[i*2+1] = label[largetst_index*2+1];
            }
            else{
                label[i*2+1] = largetst_index;
            }
            cur = label[i*2+1];
        }
        if(label[i*2]!=-1){
            cur = label[i*2];
        while(cur != -1 and label[cur*2]!=-1)
        {
            
            int largetst_index = cur;
            for(int j =0;j<12;++j){
                int q = adjacency[cur*12+j];
                
                if(q==-1){
                    continue;
                }
                if(lowGradientIndices[q]==1){
                    continue;
                }
                if((data[q]<data[largetst_index] or (data[q]==data[largetst_index] and q<largetst_index))){
                    
                    
                    largetst_index = q;
                    // }
                    
                };
            };
            if(label[largetst_index*2]!=-1){
                label[i*2] = label[largetst_index*2];
            }
            else{
                label[i*2] = largetst_index;
            }
            cur = label[i*2];
        }
        }
        
    }
        
    }
};


// std::vector<std::pair<int, int>> or_label(size2, std::make_pair(-1, -1));
std::vector<int> dec_label;
std::vector<int> or_label;
// std::vector<std::pair<int, int>> dec_label(size2, std::make_pair(-1, -1));
// bool isNumberInArray(const std::vector<int>& array, int number) {
//     for (int num : array) {
//         if (num == number) {
//             return true;  // 找到了，返回 true
//         }
//     }
//     return false;  // 遍历完毕，未找到，返回 false
// }

// int fix_maxi_critical(int index, int direction){
    
//     if (direction == 0){
        
//         if (or_direction_as[index]!=-1){
            
//             int next_vertex = from_direction_to_index1(index,or_direction_as[index]);
//             int smallest_vertex = next_vertex;
//             double threshold = std::numeric_limits<double>::lowest();
            
            
//             for(int i:adjacency[index]){
//                 if(lowGradientIndices[i] == 1){
//                     continue;
//                 }
//                 if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
//                     smallest_vertex = i;
//                     threshold = input_data[i];
//                 }
//             }
            
//             threshold = decp_data[smallest_vertex];
//             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             double d = (decp_data[index] - input_data[index] + bound )/2.0;
//             double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
//             double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            
//             if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
//                 de_direction_as[index]=or_direction_as[index];
            
//                 return 0;
//             }
            
//             if(d>=1e-16 or d1>=1e-16){
                
//                 if(decp_data[index]==decp_data[next_vertex])
//                     {
                        
//                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
//                             d/=2;
//                         }
//                         if (abs(input_data[index]-decp_data[index]+d)<=bound){
//                             decp_data[index] -= d/64;
//                         }
//                     }
//                 else{
//                     if(decp_data[index]>=decp_data[next_vertex]){
                        
//                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
//                                 d/=2;
//                         }
                        
//                         if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
//                             while(decp_data[index] - d < threshold and d>=2e-16)
//                             {
//                                 d/=2;
//                             }
                            
                            
//                         }
//                         // else if(threshold>decp_data[next_vertex]){
                            
                            
//                         //     double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/64;
                            
//                         //     if(diff2>=1e-16){
//                         //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
//                         //             diff2/=2;
//                         //         }
                                
//                         //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
//                         //             if(smallest_vertex==66783){std::cout<<"在这里11."<<std::endl;}
//                         //             decp_data[smallest_vertex]-=diff2;
//                         //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
//                         //         }
                                
                                
//                         //     }
                            
//                         // }

//                         if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-16){
//                             // if(index==1620477){
//                             //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
//                             //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
//                             //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
//                             // }
                            
//                             decp_data[index] -= d;
                            
                            
                                            
//                         }
//                         // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
//                         //     // if(index==1620477){
//                         //     //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
//                         //     //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
//                         //     //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
//                         //     // }
                            
//                         //     decp_data[next_vertex] += d1;
                            
                            
                                            
//                         // }
                        
                        
                   
//                 };
//                      }
            
                 
            
                
//             }
//             else{
                
//                 if(decp_data[index]>decp_data[next_vertex]){
//                     if(abs(input_data[index]-decp_data[next_vertex])<bound){
//                             double t = (decp_data[next_vertex]-(input_data[index]-bound))/2.0;
//                             decp_data[index] = decp_data[next_vertex] + t;
//                             // decp_data[next_vertex] = t;
//                         }
//                     else{
//                         decp_data[index] = input_data[index] - bound;
//                         // decp_data[next_vertex] = input_data[next_vertex] + bound;
//                     }
                    
//                 }
//                 else if(decp_data[index]==decp_data[next_vertex]){
//                     double d = bound - (input_data[index]-decp_data[index])/64;
//                     // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
//                     //         d/=2;
//                     // }
//                     // if(index==157569){
//                     //     std::cout<<"在这时候d: "<<d<<std::endl;
//                     // }   
//                     // double d = 1e-16;
//                     if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
//                         decp_data[index]-=d;
//                     }
//                     else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d)<=bound){
//                         // if(next_vertex==78){std::cout<<"在这里21"<<std::endl;}
//                         decp_data[next_vertex]+=d;
//                     }
//                 }
                
//             }
            
            
        
//         }
//         else{
//             // if(index==25026 and count_f_max<=770){
//             //     std::cout<<"在这里"<<std::endl;
//             // }
            
//             int largest_index = from_direction_to_index1(index,de_direction_as[index]);
//             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
//             // if(index==25026 and count_f_max<=770){
//             //     std::cout<<"改变前"<<std::endl;
//             //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
//             //     std::cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<std::endl;
//             //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
//             //     std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
//             //     std::cout<<or_direction_as[25026]<<de_direction_as[25026]<<std::endl;
//             // }
//             if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
//                 de_direction_as[index] = -1;
//             }
//             if(d>=1e-16){
                
//                 if (decp_data[index]<=decp_data[largest_index]){
//                     if(abs(input_data[largest_index]-decp_data[index]+d)){
//                         // if(largest_index==66783){std::cout<<"在这里17"<<std::endl;}
//                         decp_data[largest_index] = decp_data[index]-d;
//                     }
//                 }
                
            
                
//             }
            
//             else{
//                 if(decp_data[index]<=decp_data[largest_index]){
                    
//                     decp_data[index] = input_data[index] + bound;
//                 }
                    
//             }

            
            
//         }
        
        
    
//     }
    
//     else{
        
//         if (or_direction_ds[index]!=-1){
//             int next_vertex= from_direction_to_index1(index,or_direction_ds[index]);
            
//             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             double d =  (bound+input_data[index]-decp_data[index])/2.0;
//             // double d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
//             double d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
//             if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
//                 de_direction_ds[index]=or_direction_ds[index];
//                 return 0;
//             }

//             // if(index == 6595 and count_f_min==5){
//             //     std::cout<<"下降："<<std::endl;
//             //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
//             //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
//             //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
//             //     std::cout<<"diff: "<<diff<<std::endl;
//             //     std::cout<<"d: "<<d<<std::endl;
//             //     std::cout<<"d1: "<<d1<<std::endl;
//             // }
            
//             if(diff>=1e-16 or d>=1e-16 or d1>=1e-16){
                
//                 if(decp_data[index]==decp_data[next_vertex]){
                    
                      
                    
//                         while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
//                             diff/=2;
//                         }
                        
//                         if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
//                             // if(index==344033 and count_f_min==2){std::cout<<"在这里22"<<d<<std::endl;}
//                             decp_data[next_vertex]= decp_data[index]-diff;
//                         }
//                         else if(d1>=1e-16){
//                             // if(index==344033 and count_f_min==2){std::cout<<"在这里23"<<d<<std::endl;}
//                             decp_data[next_vertex]-=d1;
//                         }
//                         else if(d>=1e-16){
//                             // if(index==344033 and count_f_min==2){std::cout<<"在这里24"<<d<<std::endl;}
//                             decp_data[index]+=d;
//                         }

                    
                    
//                 }
//                 else{
//                     if(decp_data[index]<=decp_data[next_vertex]){
                        
//                             while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
//                                     diff/=2;
//                             }
                            
                            
//                             if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-16){
//                                 // while(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff<1e-17){
//                                 //     diff*=2;
//                                 // }
//                                 // if(index==270808 and count_f_min==1){std::cout<<"在这里2！"<< std::endl;}
//                                 while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
//                                     diff*=2;
//                                 }
//                                 if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
//                                     decp_data[next_vertex] = decp_data[index]-diff;
//                                 }
//                                 // if(index == 6595 and count_f_min==5){
//                                 //     std::cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;

//                                 // }
//                                 // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
//                                 // decp_data[next_vertex] = decp_data[index]-diff;
//                                 // if(index==89797){
//                                 //         std::cout<<"在这里2"<<diff<<", "<<d<<std::endl;
//                                 // }

//                                 // decp_data[index]+=d;
//                             }
//                             // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
//                             //     if(index==135569){std::cout<<"在这里23"<<std::endl;}
//                             //     decp_data[index]+=d;
//                             // }
//                             else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
//                                 while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-17){
//                                     d1*=2;
//                                 }
//                                 // if(count_f_min<=12){std::cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< std::endl;}
//                                 if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-16){
//                                     decp_data[next_vertex]-=d1;
//                                 }
//                                 // else{
//                                 //     decp_data[index] += d;
//                                 // }
//                                 // else{
//                                 // decp_data[next_vertex] = input_data[next_vertex] - bound;}
                                
//                             }
//                             else{
//                                 decp_data[next_vertex] = input_data[next_vertex] - bound;
//                                 //if(index == 6595 and count_f_min==5){std::cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< std::endl;}
//                             }
                            
                            
                        
                        
//                 };

//                 }
                
                

                
//             }

//             else{
                
//                 if(decp_data[index]<decp_data[next_vertex]){
//                     // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
//                     //     std::cout<<"np下降："<<std::endl;
//                     //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
//                     //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
//                     //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
//                     //     std::cout<<"diff: "<<diff<<std::endl;
//                     //     std::cout<<"d: "<<d<<std::endl;
                
//                     //     }
                        
//                         // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
//                         //     double t = decp_data[index];
//                         //     decp_data[index] = decp_data[next_vertex];
//                         //     if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
//                         //     decp_data[next_vertex] = t;
                            
//                         // }
//                         if(abs(input_data[next_vertex]-decp_data[index])<bound){
//                             double t = (decp_data[index]-(input_data[index]-bound))/2.0;
//                             // if(index==949999){std::cout<<"在这里24"<<std::endl;}
//                             // decp_data[index] = decp_data[next_vertex];
//                             // if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
//                             decp_data[next_vertex] = decp_data[index]-t;
                            
//                         }
//                         else{
//                             //if(index==949999){std::cout<<"在这里29"<<std::endl;}
//                             decp_data[index] = input_data[index] + bound;
//                         }
//                 }
                
//                 else if(decp_data[index]==decp_data[next_vertex]){
//                     double d = (bound - (input_data[index]-decp_data[index]))/64;
//                     // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
//                     //         d/=2;
//                     // }
//                     // if(index==949999){
//                     //     std::cout<<"在这里99 "<<d<<std::endl;
//                     // }   
//                     // double d = 1e-16;
//                     if(abs(input_data[index]-decp_data[index]-d)<=bound){
//                         decp_data[index]+=d;
//                     }
//                     else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
//                         // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
//                         decp_data[next_vertex]-=d;
//                     }
//                 }
//             }
            

            
            
            
//         // if(index == 6595 and count_f_min==5){
//         //         std::cout<<"下降后："<<std::endl;
//         //         std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
//         //         std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
//         //         std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
//         //         std::cout<<"diff: "<<diff<<std::endl;
//         //         std::cout<<"d: "<<d<<std::endl;
//         //         std::cout<<"d1: "<<d1<<std::endl;
//         //         std::cout<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;
//         //     }
            
        
//         }
    
//         else{
            
//             int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
//             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
//             // if(count_f_min==84){
//             //     std::cout<<"np下降："<<std::endl;
//             //     std::cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<std::endl;
//             //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
//             //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<std::endl;
//             //     std::cout<<"diff: "<<diff<<std::endl;
//             //     std::cout<<"d: "<<d<<std::endl;
                
//             // }
//             if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
//                 de_direction_ds[index] = -1;
//                 return 0;
//             }
            
//             if (diff>=1e-16){
//                 if (decp_data[index]>=decp_data[largest_index]){
//                     while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
//                         diff/=2;
//                     }
                    
                    
//                     if(abs(input_data[index]-decp_data[index]+diff)<=bound){
//                         // if(index==999973){
//                         //     std::cout<<"在这里2！"<<std::endl;
//                         // }
                        
//                         decp_data[index] -= diff;
//                     }
                    
                    
//                 }                    
//             }
            
                    
//             else{
//                 if (decp_data[index]>=decp_data[largest_index]){
                    
//                     // if(index==66783){std::cout<<"在这里15"<<std::endl;}
//                     decp_data[index] = input_data[index] - bound;
//                 }   
    
//             }


               
//         }

        
//     }    
//     return 0;
// };
void fix_maxi_critical(int index, int direction){
    int index_f = blockIdx.x * blockDim.x + threadIdx.x;
    
    
        
    double delta;
    
    int next_vertex;
   
    if (direction == 0 && lowGradientIndices[index]==0){
        
        
        // printf("%.17lf %d\n",decp_data[index],index);
        if (or_direction_as[index]!=-1){
            // printf("%d\n",index);
            // find_direction2(1,index);
            
            next_vertex = from_direction_to_index1(index,or_direction_as[index]);
            
            int smallest_vertex = next_vertex;
            double threshold = -DBL_MAX;
            
            
            for(int j=0;j<12;j++){
                int i = adjacency[index*12+j];
                if(i==-1){
                    break;
                }
                if(lowGradientIndices[i]==1){
                    continue;
                }
                
                if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];
            // 对的
            // d是把index还要降低
            // 如果是tthresh的话，那它的下限就是：input_data[index]-(abs(inaput_data[index]-decp_data[index]))
            // 之前的
            // double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double d = (decp_data[index] - input_data[index] + bound )/2.0;
            // // double d = (decp_data[index]-(input_data[index]-(abs(input_data[index]-decp_data_copy[index]))))/2.0;
            // double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            // double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d = (decp_data[index] - input_data[index] + bound )/2.0;
            double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            
            // double diff = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double d = (decp_data[index] - input_data[index] + (input_data[index]-decp_data_copy[index]))/2.0;
            // double d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // double diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // if(count_f_max==1){
            //     printf("改变后");
            //     printf("%d, %.17lf\n", index, decp_data[index]);
            //     printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
            //     printf("%.17lf %.17lf \n",d1, d);
            //     printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
            //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            // }
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
                // de_direction_as[index]=or_direction_as[index];
            
                return;
            }
            
            if(d>=1e-16 ){
                
                if(abs(decp_data[index]-decp_data[next_vertex])<1e-16)
                    {
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
                            d/=2;
                        }
                        if (abs(input_data[index]-decp_data[index]+d)<=bound){
                            delta = -d;
                            double oldValue = d_deltaBuffer[index];
                        
                        if (delta > oldValue) {
                                swap(index, delta);
                            }

                        // printf("%.17lf", delta);
                            // d_stacks[index].push(delta);
                        
                        }
                    }
                else{
                    if(decp_data[index]>=decp_data[next_vertex]){
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
                                d/=2;
                        }
                        
                        if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
                            while(decp_data[index] - d < threshold and d>=2e-16)
                            {
                                d/=2;
                            }
                            
                            
                        }
                        // else if(threshold>decp_data[next_vertex]){
                            
                            
                        //     double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/64;
                            
                        //     if(diff2>=1e-16){
                        //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
                        //             diff2/=2;
                        //         }
                                
                        //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                        //             if(smallest_vertex==66783){cout<<"在这里11."<<endl;}
                        //             decp_data[smallest_vertex]-=diff2;
                        //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                        //         }
                                
                                
                        //     }
                            
                        // }

                        if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-16){
                            // if(index==1620477){
                            //     // cout<<"next_vertex: "<<decp_data[next_vertex]<<endl;
                            //     // cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<endl;
                            //     cout<<"before index: "<<decp_data[index]<<endl;
                                
                            // }
                            
                            // decp_data[index] -= d;
                            delta = -d;
                            // int idx = atomicAdd(&number_array[index], 1);
            
                            // d_deltaBuffer1[12*index+idx] = delta;

                            double oldValue = d_deltaBuffer[index];
                            
                            if (delta > oldValue) {
                                    
                                    swap(index, delta);
                                    
                                } 
                            // d_stacks[index].push(delta);
                                            
                        }
                        // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
                        //     // if(index==1620477){
                        //     //     // cout<<"next_vertex: "<<decp_data[next_vertex]<<endl;
                        //     //     // cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<endl;
                        //     //     cout<<"before index: "<<decp_data[index]<<endl;
                                
                        //     // }
                            
                        //     decp_data[next_vertex] += d1;
                            
                            
                                            
                        // }
                        
                        // if(count_f_max==1){
                        //     printf("改变后dd");
                        //     printf("%d, %.17lf\n", index, decp_data[index]);
                        //     printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
                        //     printf("%.17lf %.17lf \n",d1, d);
                        //     printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
                        //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
                        //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
                        //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
                        //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
                        // }
                   
                };
                     }
            
                 
            
                
            }
            else{
                
                if(decp_data[index]>decp_data[next_vertex]){
                    double t = (decp_data[next_vertex]-(input_data[index]-bound))/2.0;
                    if(abs(input_data[index]-decp_data[next_vertex]+t)<=bound and t>=1e-16){
                            
                            
                            // decp_data[index] = decp_data[next_vertex] - t;
                            // decp_data[next_vertex] = t;
                            delta = decp_data[index]-(decp_data[next_vertex] - t);
                            // int oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                            
                            double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            // d_stacks.push(delta);
                            // id_stacks.push(index);
                            //printf("ok");
                            if (delta > oldValue) {
                                    swap(index, delta);
                                    
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                           
                        }
                    else{
                        
                        // decp_data[index] = input_data[index] - bound;
                        delta = decp_data[index]-(input_data[index] - bound);
                       
                        
                            double oldValue = d_deltaBuffer[index];
                        // // double expected = oldValue;
                            
                            
                        // // if(delta>10000){
                        // //         printf("chuchuo");
                        // //     }
                        
                        // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        //printf("ok");
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                
                                swap(index,delta);
                                
                                
                            }
                        // d_stacks[index].push(delta);
                        
                    }
                    // if(count_f_max==1){
                    //         printf("改变后dd");
                    //         printf("%d, %.17lf, %.17lf\n", index, decp_data[index],input_data[index]-bound);
                    //         printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
                    //         printf("%.17lf %.17lf \n",d1, d);
                    //         printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
                    //         // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
                    //         // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
                    //         // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
                    //         // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
                    //     }
                }
                else if(abs(decp_data[index]-decp_data[next_vertex])<1e-16){
                    // double bound1 = abs(input_data[index]-decp_data[index]);
                    //
                    double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    double d1 = (bound - (input_data[next_vertex]-decp_data[next_vertex]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==157569){
                    //     cout<<"在这时候d: "<<d<<endl;
                    // }   
                    // double d = 1e-16;
                    if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
                        // decp_data[index]-=d;
                        delta = -d;
                        
                        
                        
                            double oldValue = d_deltaBuffer[index];
                        
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        //printf("ok");
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(index, delta);
                                
                            }
                        // d_stacks[index].push(delta);
                        
                    }
                    
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d1)<=bound){
                        // if(next_vertex==78){cout<<"在这里21"<<endl;}
                        // decp_data[next_vertex]+=d1;
                        delta = d1;
                        
                        
                        double oldValue = d_deltaBuffer[next_vertex];
                        // // double expected = oldValue;
                            
                            
                        // // if(delta>10000){
                        // //         printf("chuchuo");
                        // //     }
                        
                        // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        //printf("ok");
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(index, delta);
                                
                            }
                        
                    }
                }
                
            }
            
            
        
        }
        else{
            // if(index==25026 and count_f_max<=770){
            //     cout<<"在这里"<<endl;
            // }
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_as[index]);
            // 对的
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // double diff = (input_data[index]-decp_data[index])/2.0;
            // double d = (input_data[largest_index]-decp_data[index])/2.0;
            // double d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // double diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // if(index==25026 and count_f_max<=770){
            //     cout<<"改变前"<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            //     cout<<or_direction_as[25026]<<de_direction_as[25026]<<endl;
            // }
            // if(count_f_max==1 and count_f_min==0){
            //     printf("fp改变后");
            //     printf("%d, %f\n", index, decp_data[index]);
            //     printf("%d %f\n", largest_index, decp_data[largest_index]);
            //     printf("%f %f \n",diff, d);
            //     printf("%.17lf %.17lf \n",input_data[index], input_data[largetst_index]);
            //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            // }
            // if(index==6345199){
            //     printf("改变后");
            //     printf("%d, %f\n", index, decp_data[index]);
            //     printf("%d %f\n",largest_index, decp_data[largest_index]);
            //     printf("%f %f \n",diff, d);
            //     printf("%d %d \n",de_direction_as[index],or_direction_as[index]);
            //     // cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     // cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     // cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            // }
            
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                return;
            }
            if(d>=1e-16){
                
                if (decp_data[index]<=decp_data[largest_index]){
                    if(abs(input_data[largest_index]-decp_data[index]+d)<bound){
                        // if(largest_index==66783){cout<<"在这里17"<<endl;}
                        // decp_data[largest_index] = decp_data[index]-d;
                        delta = (decp_data[index]-d)-decp_data[largest_index];
                        
                        
                        
                        double oldValue = d_deltaBuffer[largest_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(largest_index);
                        //printf("ok");
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(largest_index, delta);
                                
                            }
                        // d_stacks[largest_index].push(delta);
                        
                    }
                }
                
            
                
            }
            
            else{
                if(decp_data[index]<=decp_data[largest_index]){
                    // if(index==78){
                    //         cout<<"在这里1"<<endl;
                    //     }
                    // decp_data[index] = input_data[index] + bound;
                    delta = (input_data[index] + bound)-decp_data[index];
                    
                        
                         double oldValue = d_deltaBuffer[index];
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        //printf("ok");
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                
                                swap(index, delta);
                                
                            }
                    // d_stacks[index].push(delta);
                }
                    
            }

            // if(index==15885 and count_f_max==7){
            //     cout<<"改变后"<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<endl;
            //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
            //     cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
            // }
            
        }
        
        
    
    }
    
    else if (direction != 0 && lowGradientIndices[index]==0){
        
        // decp_data[index] = -1;
        // return ;
        if (or_direction_ds[index]!=-1){
            // find_direction2(1,index);
            
            int next_vertex= from_direction_to_index1(index,or_direction_ds[index]);
            
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d =  (bound+input_data[index]-decp_data[index])/2.0;
            // double d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
            double d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                // de_direction_ds[index]=or_direction_ds[index];
                return;
            }

            // if(index == 6595 and count_f_min==5){
            //     cout<<"下降："<<endl;
            //     cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
            //     cout<<"diff: "<<diff<<endl;
            //     cout<<"d: "<<d<<endl;
            //     cout<<"d1: "<<d1<<endl;
            // }
            
            
            if(diff>=1e-16){
                
                if(abs(decp_data[index]-decp_data[next_vertex])<1e-16){
                    
                      
                    
                        while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
                            diff/=2;
                        }
                        
                        if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
                            // if(index==344033 and count_f_min==2){cout<<"在这里22"<<d<<endl;}
                            // decp_data[next_vertex]= decp_data[index]-diff;
                            delta = (decp_data[index]-diff) - decp_data[next_vertex];
                            
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                
                                swap(next_vertex, delta);
                                
                            }
                            // d_stacks[next_vertex].push(delta);
                        }
                        else if(d1>=1e-16){
                            // if(index==344033 and count_f_min==2){cout<<"在这里23"<<d<<endl;}
                            delta = -d1;
                    
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                
                                swap(next_vertex,delta);
                            }
                        }
                        else if(d>=1e-16){
                            // if(index==344033 and count_f_min==2){cout<<"在这里24"<<d<<endl;}
                            
                            delta = d;
                            
                            double oldValue = d_deltaBuffer[index];
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        if (delta > oldValue) {
                                
                                
                                swap(index, delta);
                            }
                            
                        }

                    
                    
                }
                else{
                    if(decp_data[index]<=decp_data[next_vertex]){
                        
                            while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
                                    diff/=2;
                            }
                            
                            
                            if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-16){
                                // while(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff<1e-17){
                                //     diff*=2;
                                // }
                                // if(index==270808 and count_f_min==1){cout<<"在这里2！"<< endl;}
                                while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
                                    diff*=2;
                                }
                                if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
                                    // decp_data[next_vertex] = decp_data[index]-diff;
                                    delta = (decp_data[index]-diff) - decp_data[next_vertex];
                                    
                                    
                                    
                                    double oldValue = d_deltaBuffer[next_vertex];
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                
                                swap(next_vertex, delta);
                                
                                
                            }
                                }
                                // if(index == 6595 and count_f_min==5){
                                //     cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<endl;

                                // }
                                // if(next_vertex==66783){cout<<"在这里13"<<endl;}
                                // decp_data[next_vertex] = decp_data[index]-diff;
                                // if(index==89797){
                                //         cout<<"在这里2"<<diff<<", "<<d<<endl;
                                // }

                                // decp_data[index]+=d;
                            }
                            // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
                            //     if(index==135569){cout<<"在这里23"<<endl;}
                            //     decp_data[index]+=d;
                            // }
                            else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
                                while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-16){
                                    d1*=2;
                                }
                                // if(count_f_min<=12){cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< endl;}
                                if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-16){
                                    // decp_data[next_vertex]-=d1;
                                    delta = -d1;
                                    
                            
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                
                                swap(next_vertex, delta);
                                
                            }
                                }
                                
                                
                            }
                            else{
                                // decp_data[next_vertex] = input_data[next_vertex] - bound;
                                delta = (input_data[next_vertex] - bound)- decp_data[next_vertex];
                                
                            
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(next_vertex, delta);
                                
                            }
                                // if(index == 6595 and count_f_min==5){cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< endl;}
                            }
                            
                            
                        
                        
                };

                }
                
                

                
            }

            else{
                
                if(decp_data[index]<decp_data[next_vertex]){
                    // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
                    //     cout<<"np下降："<<endl;
                    //     cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
                    //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
                    //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
                    //     cout<<"diff: "<<diff<<endl;
                    //     cout<<"d: "<<d<<endl;
                
                    //     }
                        
                        // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
                        //     double t = decp_data[index];
                        //     decp_data[index] = decp_data[next_vertex];
                        //     if(next_vertex==66783){cout<<"在这里14"<<endl;}
                        //     decp_data[next_vertex] = t;
                            
                        // }
                        double t = (decp_data[index]-(input_data[index]-bound))/2.0;
                        if(abs(input_data[next_vertex]-decp_data[index]+t)<bound and t>=1e-16){
                            
                            // if(index==949999){cout<<"在这里24"<<endl;}
                            // decp_data[index] = decp_data[next_vertex];
                            // if(next_vertex==66783){cout<<"在这里14"<<endl;}
                            // decp_data[next_vertex] = decp_data[index]-t;
                            delta = (decp_data[index]-t) - decp_data[next_vertex];
                            
                            
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                            
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(next_vertex, delta);
                                
                            }
                            
                        }
                        else{
                            // if(index==949999){cout<<"在这里29"<<endl;}
                            // decp_data[index] = input_data[index] + bound;
                            delta = (input_data[index] + bound) - decp_data[index];
                            
                            double oldValue = d_deltaBuffer[index];
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(index, delta);
                                
                            }
                            
                        }
                }
                
                else if(decp_data[index]==decp_data[next_vertex]){
                    double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==949999){
                    //     cout<<"在这里99 "<<d<<endl;
                    // }   
                    // double d = 1e-16;
                    if(abs(input_data[index]-decp_data[index]-d)<=bound){
                        // decp_data[index]+=d;
                        delta = d;
                        
                        
                            double oldValue = d_deltaBuffer[index];
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(index, delta);
                                
                            }
                        // d_stacks[index].push(delta);
                    }
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
                        // if(next_vertex==66783){cout<<"在这里13"<<endl;}
                        // decp_data[next_vertex]-=d;
                        delta = -d;
                        
                        
                        double oldValue = d_deltaBuffer[next_vertex];
                        // d_stacks.push(delta);
                        // id_stacks.push(next_vertex);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(next_vertex, delta);
                                
                            }
                    }
                }
            }
            

            
            
            
        // if(index == 6595 and count_f_min==5){
        //         cout<<"下降后："<<endl;
        //         cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
        //         cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
        //         cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
        //         cout<<"diff: "<<diff<<endl;
        //         cout<<"d: "<<d<<endl;
        //         cout<<"d1: "<<d1<<endl;
        //         cout<<input_data[index]<<","<<input_data[next_vertex]<<endl;
        //     }
            
        
        }
    
        else{
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // if(count_f_min==84){
            //     cout<<"np下降："<<endl;
            //     cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<endl;
            //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
            //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<endl;
            //     cout<<"diff: "<<diff<<endl;
            //     cout<<"d: "<<d<<endl;
                
            // }
         
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                // de_direction_ds[index] = -1;
                return;
            }
            
            if (diff>=1e-16){
                if (decp_data[index]>=decp_data[largest_index]){
                    while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
                        diff/=2;
                    }
                    
                    
                    if(abs(input_data[index]-decp_data[index]+diff)<=bound){
                        // if(index==999973){
                        //     cout<<"在这里2！"<<endl;
                        // }
                        
                        // decp_data[index] -= diff;
                        delta = -diff;
                        
                        
                            double oldValue = d_deltaBuffer[index];
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(index, delta);
                                
                            }
                        // d_stacks[index].push(delta);
                    }
                    
                    
                }                    
            }
            
                    
            else{
                if (decp_data[index]>=decp_data[largest_index]){
                    
                    // if(index==66783){cout<<"在这里15"<<endl;}
                    // decp_data[index] = input_data[index] - bound;
                    delta = ((input_data[index] - bound) - decp_data[index]);
                    
                   
                            double oldValue = d_deltaBuffer[index];
                        // d_stacks.push(delta);
                        // id_stacks.push(index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(index, delta);
                                
                            }
                    // d_stacks[index].push(delta);
                }   
                
    
            }


               
        }

        
    }    
    

    

    return;
}
void fix_maxi_critical2(int index, int direction){
    
     
    double delta;
    int next_vertex;
    if (direction == 0 && lowGradientIndices[index]==0){
        
        // printf("%.17lf %d\n",decp_data[index],index);
        if (or_direction_as[index]!=-1){
            // printf("%d\n",index);
            // find_direction2(1,index);
            
            next_vertex = from_direction_to_index1(index,or_direction_as[index]);
            
            int smallest_vertex = next_vertex;
            double threshold = std::numeric_limits<double>::lowest();
            
            
            for(int j=0;j<12;j++){
                int i = adjacency[index*12+j];
                if(i==-1){
                    break;
                }
                if(lowGradientIndices[i]==1){
                    continue;
                }
                
                if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];
            // 对的
            // d是把index还要降低
            // 如果是tthresh的话，那它的下限就是：input_data[index]-(abs(inaput_data[index]-decp_data[index]))
            // 之前的
            // double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double d = (decp_data[index] - input_data[index] + bound )/2.0;
            // // double d = (decp_data[index]-(input_data[index]-(abs(input_data[index]-decp_data_copy[index]))))/2.0;
            // double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            // double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d = (decp_data[index] - input_data[index] + bound )/2.0;
            double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            
            // double diff = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double d = (decp_data[index] - input_data[index] + (input_data[index]-decp_data_copy[index]))/2.0;
            // double d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // double diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // if(count_f_max==1){
            //     printf("改变后");
            //     printf("%d, %.17lf\n", index, decp_data[index]);
            //     printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
            //     printf("%.17lf %.17lf \n",d1, d);
            //     printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
            //     // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
                // de_direction_as[index]=or_direction_as[index];
            
                return;
            }
            
            if(d>=1e-16 ){
                
                if(abs(decp_data[index]-decp_data[next_vertex])<1e-16)
                    {
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
                            d/=2;
                        }
                        if (abs(input_data[index]-decp_data[index]+d)<=bound){
                            delta = -d;
                            double oldValue = d_deltaBuffer[index];
                            // int expected = d_deltaBuffer[index];
                        
                      
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                swap(index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                        
                        }
                    }
                else{
                    if(decp_data[index]>=decp_data[next_vertex]){
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
                                d/=2;
                        }
                        
                        if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
                            while(decp_data[index] - d < threshold and d>=2e-16)
                            {
                                d/=2;
                            }
                            
                            
                        }
                        // else if(threshold>decp_data[next_vertex]){
                            
                            
                        //     double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/64;
                            
                        //     if(diff2>=1e-16){
                        //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
                        //             diff2/=2;
                        //         }
                                
                        //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                        //             if(smallest_vertex==66783){std::cout<<"在这里11."<<std::endl;}
                        //             decp_data[smallest_vertex]-=diff2;
                        //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
                        //         }
                                
                                
                        //     }
                            
                        // }

                        if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-16){
                            // if(index==1620477){
                            //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
                            //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
                            //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
                            // }
                            
                            // decp_data[index] -= d;
                            delta = -d;
                            
                            double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    
                                    swap(index,delta);
                                    
                                    
                                } 
                                            
                        }
                        // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
                        //     // if(index==1620477){
                        //     //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
                        //     //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
                        //     //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
                        //     // }
                            
                        //     decp_data[next_vertex] += d1;
                            
                            
                                            
                        // }
                        
                        // if(count_f_max==1){
                        //     printf("改变后dd");
                        //     printf("%d, %.17lf\n", index, decp_data[index]);
                        //     printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
                        //     printf("%.17lf %.17lf \n",d1, d);
                        //     printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
                        //     // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
                        //     // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
                        //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
                        //     // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
                        // }
                   
                };
                     }
            
                 
            
                
            }
            else{
                
                if(decp_data[index]>decp_data[next_vertex]){
                    double t = (decp_data[next_vertex]-(input_data[index]-bound))/2.0;
                    if(abs(input_data[index]-decp_data[next_vertex]+t)<=bound and t>=1e-16){
                            
                            
                            // decp_data[index] = decp_data[next_vertex] - t;
                            // decp_data[next_vertex] = t;
                            delta = (decp_data[next_vertex] - t) - decp_data[index];
                            // double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                            
                            double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(index,delta);
                                    
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                           
                        }
                    else{
                        
                        // decp_data[index] = input_data[index] - bound;
                        delta = (input_data[index] - bound) - decp_data[index];

                        double oldValue = d_deltaBuffer[index];
                        // double expected = oldValue;
                            
                            
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(index,delta);
                                
                            }
                        
                    }
                    // if(count_f_max==1){
                    //         printf("改变后dd");
                    //         printf("%d, %.17lf, %.17lf\n", index, decp_data[index],input_data[index]-bound);
                    //         printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
                    //         printf("%.17lf %.17lf \n",d1, d);
                    //         printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
                    //         // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
                    //         // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
                    //         // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
                    //         // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
                    //     }
                }
                else if(decp_data[index]==decp_data[next_vertex]){
                    // double bound1 = abs(input_data[index]-decp_data[index]);
                    //
                    double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    double d1 = (bound - (input_data[next_vertex]-decp_data[next_vertex]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==157569){
                    //     std::cout<<"在这时候d: "<<d<<std::endl;
                    // }   
                    // double d = 1e-16;
                    if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
                        // decp_data[index]-=d;
                        delta = -d;
                        double oldValue = d_deltaBuffer[index];
                        // double expected = oldValue;
                            
                            
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(index,delta);
                                
                            }
                        
                    }
                    
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d1)<=bound){
                        // if(next_vertex==78){std::cout<<"在这里21"<<std::endl;}
                        // decp_data[next_vertex]+=d1;
                        delta = d1;
                        double oldValue = d_deltaBuffer[next_vertex];
                        // double expected = oldValue;
                            
                            
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(next_vertex,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                        
                    }
                }
                
            }
            
            
        
        }
        else{
            // if(index==25026 and count_f_max<=770){
            //     std::cout<<"在这里"<<std::endl;
            // }
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_as[index]);
            // 对的
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // double diff = (input_data[index]-decp_data[index])/2.0;
            // double d = (input_data[largest_index]-decp_data[index])/2.0;
            // double d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // double diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // if(index==25026 and count_f_max<=770){
            //     std::cout<<"改变前"<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            //     std::cout<<or_direction_as[25026]<<de_direction_as[25026]<<std::endl;
            // }
            // if(count_f_max==1 and count_f_min==0){
            //     printf("fp改变后");
            //     printf("%d, %f\n", index, decp_data[index]);
            //     printf("%d %f\n", largest_index, decp_data[largest_index]);
            //     printf("%f %f \n",diff, d);
            //     printf("%.17lf %.17lf \n",input_data[index], input_data[largetst_index]);
            //     // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            // if(index==6345199){
            //     printf("改变后");
            //     printf("%d, %f\n", index, decp_data[index]);
            //     printf("%d %f\n",largest_index, decp_data[largest_index]);
            //     printf("%f %f \n",diff, d);
            //     printf("%d %d \n",de_direction_as[index],or_direction_as[index]);
            //     // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                de_direction_as[index] = -1;
            }
            if(d>=1e-16){
                
                if (decp_data[index]<=decp_data[largest_index]){
                    if(abs(input_data[largest_index]-decp_data[index]+d)<bound){
                        // if(largest_index==66783){std::cout<<"在这里17"<<std::endl;}
                        // decp_data[largest_index] = decp_data[index]-d;
                        delta = (decp_data[index]-d)-decp_data[largest_index];
                        double oldValue = d_deltaBuffer[largest_index];
                        // double expected = oldValue;
                            
                            
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(largest_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                        
                    }
                }
                
            
                
            }
            
            else{
                if(decp_data[index]<=decp_data[largest_index]){
                    // if(index==78){
                    //         std::cout<<"在这里1"<<std::endl;
                    //     }
                    // decp_data[index] = input_data[index] + bound;
                    delta = (input_data[index] + bound)-decp_data[index];
                    double oldValue = d_deltaBuffer[index];
                        // double expected = oldValue;
                            
                            
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                    
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                    if (delta > oldValue) {
                            swap(index,delta);
                            
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                        }
                }
                    
            }

            // if(index==15885 and count_f_max==7){
            //     std::cout<<"改变后"<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            
        }
        
        
    
    }
    
    else if (direction != 0 && lowGradientIndices[index]==0){
        if (or_direction_ds[index]!=-1){
            // find_direction2(1,index);
            int next_vertex= from_direction_to_index1(index,or_direction_ds[index]);
            
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d =  (bound+input_data[index]-decp_data[index])/2.0;
            // double d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
            double d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                de_direction_ds[index]=or_direction_ds[index];
                return;
            }

            // if(index == 6595 and count_f_min==5){
            //     std::cout<<"下降："<<std::endl;
            //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
            //     std::cout<<"diff: "<<diff<<std::endl;
            //     std::cout<<"d: "<<d<<std::endl;
            //     std::cout<<"d1: "<<d1<<std::endl;
            // }
            
            
            if(diff>=1e-16){
                
                if(decp_data[index]==decp_data[next_vertex]){
                    
                      
                    
                        while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
                            diff/=2;
                        }
                        
                        if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
                            // if(index==344033 and count_f_min==2){std::cout<<"在这里22"<<d<<std::endl;}
                            // decp_data[next_vertex]= decp_data[index]-diff;
                            delta = (decp_data[index]-diff) - decp_data[next_vertex];
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(next_vertex,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                        }
                        else if(d1>=1e-16){
                            // if(index==344033 and count_f_min==2){std::cout<<"在这里23"<<d<<std::endl;}
                            delta = -d1;
                            double oldValue = d_deltaBuffer[next_vertex];
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[next_vertex], delta);
                            // if(delta>10000){
                    //         printf("chuchuo");
                    //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                swap(next_vertex,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                        }
                        else if(d>=1e-16){
                            // if(index==344033 and count_f_min==2){std::cout<<"在这里24"<<d<<std::endl;}
                            
                            delta = d;
                            
                            double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(index,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                        }

                    
                    
                }
                else{
                    if(decp_data[index]<=decp_data[next_vertex]){
                        
                            while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
                                    diff/=2;
                            }
                            
                            
                            if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-16){
                                // while(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff<1e-17){
                                //     diff*=2;
                                // }
                                // if(index==270808 and count_f_min==1){std::cout<<"在这里2！"<< std::endl;}
                                while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
                                    diff*=2;
                                }
                                if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
                                    // decp_data[next_vertex] = decp_data[index]-diff;
                                    delta = (decp_data[index]-diff) - decp_data[next_vertex];
                                    
                                    double oldValue = d_deltaBuffer[next_vertex];
                                    // double expected = oldValue;
                                        
                                        
                                    // if(delta>10000){
                                    //         printf("chuchuo");
                                    //     }
                                    // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                                    if (delta > oldValue) {
                                            swap(next_vertex,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                        }
                                }
                                // if(index == 6595 and count_f_min==5){
                                //     std::cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;

                                // }
                                // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
                                // decp_data[next_vertex] = decp_data[index]-diff;
                                // if(index==89797){
                                //         std::cout<<"在这里2"<<diff<<", "<<d<<std::endl;
                                // }

                                // decp_data[index]+=d;
                            }
                            // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
                            //     if(index==135569){std::cout<<"在这里23"<<std::endl;}
                            //     decp_data[index]+=d;
                            // }
                            else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
                                while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-16){
                                    d1*=2;
                                }
                                // if(count_f_min<=12){std::cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< std::endl;}
                                if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-16){
                                    // decp_data[next_vertex]-=d1;
                                    delta = -d1;
                                    
                            double oldValue = d_deltaBuffer[next_vertex];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(next_vertex,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                                }
                                // else{
                                //     decp_data[index] += d;
                                // }
                                // else{
                                // decp_data[next_vertex] = input_data[next_vertex] - bound;}
                                
                            }
                            else{
                                // decp_data[next_vertex] = input_data[next_vertex] - bound;
                                delta = (input_data[next_vertex] - bound)- decp_data[next_vertex];
                                
                            double oldValue = d_deltaBuffer[next_vertex];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(next_vertex,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                                // if(index == 6595 and count_f_min==5){std::cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< std::endl;}
                            }
                            
                            
                        
                        
                };

                }
                
                

                
            }

            else{
                
                if(decp_data[index]<decp_data[next_vertex]){
                    // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
                    //     std::cout<<"np下降："<<std::endl;
                    //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
                    //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
                    //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
                    //     std::cout<<"diff: "<<diff<<std::endl;
                    //     std::cout<<"d: "<<d<<std::endl;
                
                    //     }
                        
                        // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
                        //     double t = decp_data[index];
                        //     decp_data[index] = decp_data[next_vertex];
                        //     if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
                        //     decp_data[next_vertex] = t;
                            
                        // }
                        double t = (decp_data[index]-(input_data[index]-bound))/2.0;
                        if(abs(input_data[next_vertex]-decp_data[index]+t)<bound and t>=1e-16){
                            
                            // if(index==949999){std::cout<<"在这里24"<<std::endl;}
                            // decp_data[index] = decp_data[next_vertex];
                            // if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
                            // decp_data[next_vertex] = decp_data[index]-t;
                            delta = (decp_data[index]-t) - decp_data[next_vertex];
                            
                            double oldValue = d_deltaBuffer[next_vertex];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(next_vertex,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                            
                        }
                        else{
                            // if(index==949999){std::cout<<"在这里29"<<std::endl;}
                            // decp_data[index] = input_data[index] + bound;
                            delta = (input_data[index] + bound) - decp_data[index];
                            
                            double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(index,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                        }
                }
                
                else if(decp_data[index]==decp_data[next_vertex]){
                    double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==949999){
                    //     std::cout<<"在这里99 "<<d<<std::endl;
                    // }   
                    // double d = 1e-16;
                    if(abs(input_data[index]-decp_data[index]-d)<=bound){
                        // decp_data[index]+=d;
                        delta = d;
                        double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(index,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                    }
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
                        // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
                        // decp_data[next_vertex]-=d;
                        delta = -d;
                        double oldValue = d_deltaBuffer[next_vertex];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(next_vertex,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                    
                                }
                    }
                }
            }
            

            
            
            
        // if(index == 6595 and count_f_min==5){
        //         std::cout<<"下降后："<<std::endl;
        //         std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
        //         std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
        //         std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
        //         std::cout<<"diff: "<<diff<<std::endl;
        //         std::cout<<"d: "<<d<<std::endl;
        //         std::cout<<"d1: "<<d1<<std::endl;
        //         std::cout<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;
        //     }
            
        
        }
    
        else{
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // if(count_f_min==84){
            //     std::cout<<"np下降："<<std::endl;
            //     std::cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<std::endl;
            //     std::cout<<"diff: "<<diff<<std::endl;
            //     std::cout<<"d: "<<d<<std::endl;
                
            // }
         
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                de_direction_ds[index] = -1;
                return;
            }
            
            if (diff>=1e-16){
                if (decp_data[index]>=decp_data[largest_index]){
                    while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
                        diff/=2;
                    }
                    
                    
                    if(abs(input_data[index]-decp_data[index]+diff)<=bound){
                        // if(index==999973){
                        //     std::cout<<"在这里2！"<<std::endl;
                        // }
                        
                        // decp_data[index] -= diff;
                        delta = -diff;
                        double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                   swap(index,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                    }
                    
                    
                }                    
            }
            
                    
            else{
                if (decp_data[index]>=decp_data[largest_index]){
                    
                    // if(index==66783){std::cout<<"在这里15"<<std::endl;}
                    // decp_data[index] = input_data[index] - bound;
                    delta = ((input_data[index] - bound) - decp_data[index]);
                    double oldValue = d_deltaBuffer[index];
                            // double expected = oldValue;
                                
                                
                            // if(delta>10000){
                            //         printf("chuchuo");
                            //     }
                            // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    swap(index,delta);
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                }
                }   
                
    
            }


               
        }

        
    }    
    

    

    return;
}
void fix_maxi_critical1(int index,int direction){
    
    
        
    
    if (direction == 0 && lowGradientIndices[index]==0){
        
        
        // printf("%d\n",index);
        if (or_direction_as[index]!=-1){
            // printf("%d\n",index);
            // find_direction2(1,index);
            int next_vertex = from_direction_to_index1(index,or_direction_as[index]);
            
            int smallest_vertex = next_vertex;
            double threshold = std::numeric_limits<double>::lowest();
            
            
            for(int j=0;j<12;j++){
                int i = adjacency[index*12+j];
                if(i==-1 or lowGradientIndices[i]==1){
                    continue;
                }
                
                
                if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];
            // 对的
            // d是把index还要降低
            // 如果是tthresh的话，那它的下限就是：input_data[index]-(abs(inaput_data[index]-decp_data[index]))
            // 之前的
            // double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double d = (decp_data[index] - input_data[index] + bound )/2.0;
            // // double d = (decp_data[index]-(input_data[index]-(abs(input_data[index]-decp_data_copy[index]))))/2.0;
            // double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            // double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d = (decp_data[index] - input_data[index] + bound )/2.0;
            double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
            double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double diff = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            // double d = (decp_data[index] - input_data[index] + (input_data[index]-decp_data_copy[index]))/2.0;
            // double d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // double diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
                de_direction_as[index]=or_direction_as[index];
            
                return;
            }
            
            if(d>=1e-16 ){
                
                if(decp_data[index]==decp_data[next_vertex])
                    {
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
                            d/=2;
                        }
                        if (abs(input_data[index]-decp_data[index]+d)<=bound){
                            decp_data[index] -= d;
                        }
                    }
                else{
                    if(decp_data[index]>=decp_data[next_vertex]){
                        
                        while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
                                d/=2;
                        }
                        
                        if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
                            while(decp_data[index] - d < threshold and d>=2e-16)
                            {
                                d/=2;
                            }
                            
                            
                        }
                        // else if(threshold>decp_data[next_vertex]){
                            
                            
                        //     double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/64;
                            
                        //     if(diff2>=1e-16){
                        //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
                        //             diff2/=2;
                        //         }
                                
                        //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                        //             if(smallest_vertex==66783){std::cout<<"在这里11."<<std::endl;}
                        //             decp_data[smallest_vertex]-=diff2;
                        //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
                        //         }
                                
                                
                        //     }
                            
                        // }

                        if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-16){
                            // if(index==1620477){
                            //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
                            //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
                            //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
                            // }
                            
                            decp_data[index] -= d;
                            
                            
                                            
                        }
                        // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
                        //     // if(index==1620477){
                        //     //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
                        //     //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
                        //     //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
                        //     // }
                            
                        //     decp_data[next_vertex] += d1;
                            
                            
                                            
                        // }
                        
                        
                   
                };
                     }
            
                 
            
                
            }
            else{
                
                if(decp_data[index]>decp_data[next_vertex]){
                    double t = (decp_data[next_vertex]-(input_data[index]-bound))/2.0;
                    if(abs(input_data[index]-decp_data[next_vertex]+t)<=bound and t>=1e-16){
                            
                            
                            decp_data[index] = decp_data[next_vertex] - t;
                            // decp_data[next_vertex] = t;
                        }
                    else{
                        
                        decp_data[index] = input_data[index] - bound;
                        
                    }
                    // if(count_f_max==1){
                    //         printf("改变后dd");
                    //         printf("%d, %.17lf, %.17lf\n", index, decp_data[index],input_data[index]-bound);
                    //         printf("%d %.17lf\n", next_vertex, decp_data[next_vertex]);
                    //         printf("%.17lf %.17lf \n",d1, d);
                    //         printf("%.17lf %.17lf \n",input_data[index], input_data[next_vertex]);
                    //         // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
                    //         // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
                    //         // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
                    //         // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
                    //     }
                }
                else if(decp_data[index]==decp_data[next_vertex]){
                    // double bound1 = abs(input_data[index]-decp_data[index]);
                    //
                    double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    double d1 = (bound - (input_data[next_vertex]-decp_data[next_vertex]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==157569){
                    //     std::cout<<"在这时候d: "<<d<<std::endl;
                    // }   
                    // double d = 1e-16;
                    if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
                        decp_data[index]-=d;
                    }
                    
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d1)<=bound){
                        // if(next_vertex==78){std::cout<<"在这里21"<<std::endl;}
                        decp_data[next_vertex]+=d1;
                    }
                }
                
            }
            
            
        
        }
        else{
            // if(index==25026 and count_f_max<=770){
            //     std::cout<<"在这里"<<std::endl;
            // }
            // find_direction2(0,index);

            int largest_index = from_direction_to_index1(index,de_direction_as[index]);
            // 对的
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // double diff = (input_data[index]-decp_data[index])/2.0;
            // double d = (input_data[largest_index]-decp_data[index])/2.0;
            // double d1 = ((input_data[next_vertex] + (input_data[next_vertex]-decp_data_copy[next_vertex])) - decp_data[next_vertex])/2.0;
            // double diff1 = ((input_data[next_vertex]-decp_data_copy[next_vertex]) - (input_data[next_vertex]-decp_data[index]))/2.0;
            
            
            // if(count_f_max==1 and count_f_min==0){
            //     printf("fp改变后");
            //     printf("%d, %f\n", index, decp_data[index]);
            //     printf("%d %f\n", largest_index, decp_data[largest_index]);
            //     printf("%f %f \n",diff, d);
            //     printf("%.17lf %.17lf \n",input_data[index], input_data[largetst_index]);
            //     // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            // if(index==6345199){
            //     printf("改变后");
            //     printf("%d, %f\n", index, decp_data[index]);
            //     printf("%d %f\n",largest_index, decp_data[largest_index]);
            //     printf("%f %f \n",diff, d);
            //     printf("%d %d \n",de_direction_as[index],or_direction_as[index]);
            //     // std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     // std::cout<<"next_vertex: "<<next_vertex<<","<<decp_data[next_vertex]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     // std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                de_direction_as[index] = -1;
            }
            if(d>=1e-16){
                
                if (decp_data[index]<=decp_data[largest_index]){
                    if(abs(input_data[largest_index]-decp_data[index]+d)){
                        // if(largest_index==66783){std::cout<<"在这里17"<<std::endl;}
                        decp_data[largest_index] = decp_data[index]-d;
                    }
                }
                
            
                
            }
            
            else{
                if(decp_data[index]<=decp_data[largest_index]){
                    // if(index==78){
                    //         std::cout<<"在这里1"<<std::endl;
                    //     }
                    decp_data[index] = input_data[index] + bound;
                }
                    
            }

            // if(index==15885 and count_f_max==7){
            //     std::cout<<"改变后"<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<std::endl;
            //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
            //     std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
            // }
            
        }
        
        
    
    }
    
    else if (direction != 0 && lowGradientIndices[index]==0){
        
        if (or_direction_ds[index]!=-1){
            // find_direction2(1,index);
            int next_vertex= from_direction_to_index1(index,or_direction_ds[index]);
            
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d =  (bound+input_data[index]-decp_data[index])/2.0;
            // double d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
            double d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                de_direction_ds[index]=or_direction_ds[index];
                return;
            }

            // if(index == 6595 and count_f_min==5){
            //     std::cout<<"下降："<<std::endl;
            //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
            //     std::cout<<"diff: "<<diff<<std::endl;
            //     std::cout<<"d: "<<d<<std::endl;
            //     std::cout<<"d1: "<<d1<<std::endl;
            // }
            
            if(diff>=1e-16){
                
                if(decp_data[index]==decp_data[next_vertex]){
                    
                      
                    
                        while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
                            diff/=2;
                        }
                        
                        if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
                            // if(index==344033 and count_f_min==2){std::cout<<"在这里22"<<d<<std::endl;}
                            decp_data[next_vertex]= decp_data[index]-diff;
                        }
                        else if(d1>=1e-16){
                            // if(index==344033 and count_f_min==2){std::cout<<"在这里23"<<d<<std::endl;}
                            decp_data[next_vertex]-=d1;
                        }
                        else if(d>=1e-16){
                            // if(index==344033 and count_f_min==2){std::cout<<"在这里24"<<d<<std::endl;}
                            decp_data[index]+=d;
                        }

                    
                    
                }
                else{
                    if(decp_data[index]<=decp_data[next_vertex]){
                        
                            while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
                                    diff/=2;
                            }
                            
                            
                            if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-16){
                                // while(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff<1e-17){
                                //     diff*=2;
                                // }
                                // if(index==270808 and count_f_min==1){std::cout<<"在这里2！"<< std::endl;}
                                while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
                                    diff*=2;
                                }
                                if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
                                    decp_data[next_vertex] = decp_data[index]-diff;
                                }
                                // if(index == 6595 and count_f_min==5){
                                //     std::cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;

                                // }
                                // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
                                // decp_data[next_vertex] = decp_data[index]-diff;
                                // if(index==89797){
                                //         std::cout<<"在这里2"<<diff<<", "<<d<<std::endl;
                                // }

                                // decp_data[index]+=d;
                            }
                            // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
                            //     if(index==135569){std::cout<<"在这里23"<<std::endl;}
                            //     decp_data[index]+=d;
                            // }
                            else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
                                while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-16){
                                    d1*=2;
                                }
                                // if(count_f_min<=12){std::cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< std::endl;}
                                if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-16){
                                    decp_data[next_vertex]-=d1;
                                }
                                // else{
                                //     decp_data[index] += d;
                                // }
                                // else{
                                // decp_data[next_vertex] = input_data[next_vertex] - bound;}
                                
                            }
                            else{
                                decp_data[next_vertex] = input_data[next_vertex] - bound;
                                // if(index == 6595 and count_f_min==5){std::cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< std::endl;}
                            }
                            
                            
                        
                        
                };

                }
                
                

                
            }

            else{
                
                if(decp_data[index]<decp_data[next_vertex]){
                    // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
                    //     std::cout<<"np下降："<<std::endl;
                    //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
                    //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
                    //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
                    //     std::cout<<"diff: "<<diff<<std::endl;
                    //     std::cout<<"d: "<<d<<std::endl;
                
                    //     }
                        
                        // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
                        //     double t = decp_data[index];
                        //     decp_data[index] = decp_data[next_vertex];
                        //     if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
                        //     decp_data[next_vertex] = t;
                            
                        // }
                        double t = (decp_data[index]-(input_data[index]-bound))/2.0;
                        if(abs(input_data[next_vertex]-decp_data[index]+t)<bound and t>=1e-16){
                            
                            // if(index==949999){std::cout<<"在这里24"<<std::endl;}
                            // decp_data[index] = decp_data[next_vertex];
                            // if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
                            decp_data[next_vertex] = decp_data[index]-t;
                            
                        }
                        else{
                            // if(index==949999){std::cout<<"在这里29"<<std::endl;}
                            decp_data[index] = input_data[index] + bound;
                        }
                }
                
                else if(decp_data[index]==decp_data[next_vertex]){
                    double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
                    //         d/=2;
                    // }
                    // if(index==949999){
                    //     std::cout<<"在这里99 "<<d<<std::endl;
                    // }   
                    // double d = 1e-16;
                    if(abs(input_data[index]-decp_data[index]-d)<=bound){
                        decp_data[index]+=d;
                    }
                    else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
                        // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
                        decp_data[next_vertex]-=d;
                    }
                }
            }
            

            
            
            
        // if(index == 6595 and count_f_min==5){
        //         std::cout<<"下降后："<<std::endl;
        //         std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
        //         std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
        //         std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
        //         std::cout<<"diff: "<<diff<<std::endl;
        //         std::cout<<"d: "<<d<<std::endl;
        //         std::cout<<"d1: "<<d1<<std::endl;
        //         std::cout<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;
        //     }
            
        
        }
    
        else{
            // find_direction2(0,index);
            int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            // if(count_f_min==84){
            //     std::cout<<"np下降："<<std::endl;
            //     std::cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<std::endl;
            //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
            //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<std::endl;
            //     std::cout<<"diff: "<<diff<<std::endl;
            //     std::cout<<"d: "<<d<<std::endl;
                
            // }
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                de_direction_ds[index] = -1;
                return;
            }
            
            if (diff>=1e-16){
                if (decp_data[index]>=decp_data[largest_index]){
                    while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
                        diff/=2;
                    }
                    
                    
                    if(abs(input_data[index]-decp_data[index]+diff)<=bound){
                        // if(index==999973){
                        //     std::cout<<"在这里2！"<<std::endl;
                        // }
                        
                        decp_data[index] -= diff;
                    }
                    
                    
                }                    
            }
            
                    
            else{
                if (decp_data[index]>=decp_data[largest_index]){
                    
                    // if(index==66783){std::cout<<"在这里15"<<std::endl;}
                    decp_data[index] = input_data[index] - bound;
                }   
    
            }


               
        }

        
    }    
    return;
}
// int fix_maxi_critical(int index, int direction){
    
//     if (direction == 0){
        
//         if (or_direction_as[index]!=-1){
            
//             int next_vertex = from_direction_to_index1(index,or_direction_as[index]);
//             int smallest_vertex = next_vertex;
//             double threshold = std::numeric_limits<double>::lowest();
            
            
//             for(int i:adjacency[index]){
//                 if(lowGradientIndices[i] == 1){
//                     continue;
//                 }
//                 if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
//                     smallest_vertex = i;
//                     threshold = input_data[i];
//                 }
//             }
//             // if(lowGradientIndices[index] == 1 or lowGradientIndices.find(next_vertex)!=lowGradientIndices.end() or lowGradientIndices.find(smallest_vertex)!=lowGradientIndices.end()){
//             //         std::cout<<index<<","<<next_vertex<<","<<smallest_vertex<<std::endl;
//             //     }
//             threshold = decp_data[smallest_vertex];
//             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             double d = (bound - (input_data[index]-decp_data[index]))/2.0;
            
//             if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
//                 de_direction_as[index]=or_direction_as[index];
            
//                 return 0;
//             }
            
//             if(d>=1e-16){
                
//                 if(decp_data[index]==decp_data[next_vertex])
//                     {
                        
//                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
//                             d/=2;
//                         }
//                         if (abs(input_data[index]-decp_data[index]+d)<=bound){
//                             decp_data[index] -= d;
//                         }

                    
//                     }
//                 else{
//                     if(decp_data[index]>=decp_data[next_vertex]){
                        
//                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
//                                 d/=2;
//                         }
                        
//                         if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
//                             while(decp_data[index] - d < threshold and d>=2e-16)
//                             {
//                                 d/=2;
//                             }
                            
                            
//                         }
//                         else if(threshold>decp_data[next_vertex]){
                            
                            
//                             double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/2;
                            
//                             if(diff2>=1e-16){
//                                 while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
//                                     diff2/=2;
//                                 }
                                
//                                 if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
//                                     decp_data[smallest_vertex]-=diff2;
//                                     // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
//                                 }
                                
                                
//                             }
                            
//                         }

//                         if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex]){
//                             decp_data[index] -= d;
                            
//                         }
                        
                        
                   
//                 };
//                      }

                 
            
                
//             }
//             else{
                
//                 if(decp_data[index]>=decp_data[next_vertex]){
//                     if(abs(input_data[index]-(input_data[next_vertex] -bound+ decp_data[index])/2.0)<=bound){
//                         decp_data[index] = (input_data[next_vertex] -bound + decp_data[index])/2.0;
//                     }
//                     else{
                        
//                         decp_data[index] = input_data[index] - bound;
//                     }
                    
//                 }
                
//             }
            
            
        
//         }
//         else{
            
//             int largest_index = from_direction_to_index1(index,de_direction_as[index]);
//             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
//             if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
//                 de_direction_as[index] = -1;
//             }
//             if(d>=1e-16){
                
//                 if (decp_data[index]<=decp_data[largest_index]){
//                     if(abs(input_data[largest_index]-decp_data[index]+d)){
//                         decp_data[largest_index] = decp_data[index]-d;
//                     }
//                 }
                
            
                
//             }
            
//             else{
//                 if(decp_data[index]<=decp_data[largest_index]){
//                     decp_data[index] = input_data[index] + bound;
//                 }
                    
//             }
            
//         }
        
        
    
//     }
    
//     else{
        
//         if (or_direction_ds[index]!=-1){
//             int next_vertex= from_direction_to_index1(index,or_direction_ds[index]);
            
//             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             double d =  (bound-(input_data[index]-decp_data[index]))/2.0;
            
            
//             if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
//                 de_direction_ds[index]=or_direction_ds[index];
//                 return 0;
//             }
            
//             if(diff>=1e-16 or d>=1e-16){
//                 if(decp_data[index]==decp_data[next_vertex]){
                    
                    
//                         while(abs(input_data[next_vertex]-decp_data[index]-d)>bound and d>=2e-16){
//                             d/=2;
//                         }
                        
//                         if(abs(input_data[index]-decp_data[index]-d)<=bound){
//                             decp_data[index]+=d;
//                         }
                    
                    
                    
                    
//                 }
//                 else{
//                     if(decp_data[index]<=decp_data[next_vertex]){
                        
//                             while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
//                                     diff/=2;
//                             }
                            
//                             if (abs(input_data[next_vertex]-decp_data[index]+d)<=bound and decp_data[index]<=decp_data[next_vertex]){
//                                 decp_data[next_vertex] = decp_data[index]-diff;
//                             }
                            
                            
                        
                        
//                 };

//                 }
                
                

                
//             }

//             else{
                
//                 if(decp_data[index]<=decp_data[next_vertex]){
//                     if(abs(input_data[index]-(input_data[next_vertex] + bound + decp_data[index])/2.0)<=bound){
//                         decp_data[index] = (input_data[next_vertex] + bound + decp_data[index])/2.0;
//                     }
//                     else{
//                         decp_data[index] = input_data[index] + bound;
//                     }
//                 }
//             }
            

            
            
            

            
        
//         }
    
//         else{
            
//             int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
//             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            
//             if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
//                 de_direction_ds[index] = -1;
//                 return 0;
//             }
            
//             if (diff>=1e-16){
//                 if (decp_data[index]>=decp_data[largest_index]){
//                     while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
//                         diff/=2;
//                     }
                    
                    
//                     if(abs(input_data[index]-decp_data[index]+diff)<=bound){
//                         decp_data[index] -= diff;
//                     }
                    
                    
//                 }                    
//             }
            
                    
//             else{
//                 if (decp_data[index]>=decp_data[largest_index]){
//                     decp_data[index] = input_data[index] - bound;
//                 }   
    
//             }


               
//         }

        
//     }    
//     return 0;
// };
// int fixpath1(int index, int direction){
    
//     if(direction==0){
//         int cur = index;
        
//         while (or_direction_as[cur] == de_direction_as[cur]){
//             int next_vertex =  from_direction_to_index1(cur,de_direction_as[cur]);
            
//             if(de_direction_as[cur]==-1 && next_vertex == cur){
//                 cur = -1;
//                 break;
//             }
//             if(next_vertex == cur){
//                 cur = next_vertex;
//                 break;
//             };
            
//             cur = next_vertex;
//         }

//         int start_vertex = cur;
        
        
//         if (start_vertex==-1) return 0;
//         else{
            
//             int false_index= from_direction_to_index1(cur,de_direction_as[cur]);
//             int true_index= from_direction_to_index1(cur, or_direction_as[cur]);
//             if(false_index==true_index) return 0;
            
//             // double diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
//             double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
//             // double d = (bound-abs(input_data[true_index]-decp_data[true_index]))/2.0;
//             double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            
//             // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            
//             if(decp_data[false_index]<decp_data[true_index]){
//                 de_direction_as[cur]=or_direction_as[cur];
//                 // 103781, 103830
//                 return 0;
//             }
//             double threshold = std::numeric_limits<double>::lowest();
//             int smallest_vertex = false_index;
            
//             for(int i:adjacency[false_index]){
//                 if(lowGradientIndices[i] == 1){
//                 continue;}
//                 if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
//                     smallest_vertex = i;
//                     threshold = input_data[i];
//                 }
//             }
            
//             threshold = decp_data[smallest_vertex];

//             double threshold1 = std::numeric_limits<double>::max();
//             int smallest_vertex1 = true_index;
            
//             for(int i:adjacency[true_index]){
//                 if(lowGradientIndices[i] == 1){
//                 continue;}
//                 if(input_data[i]>input_data[true_index] and input_data[i]<threshold1 and i!=true_index){
//                     smallest_vertex1 = i;
//                     threshold = input_data[i];
//                 }
//             }
            
//             threshold1 = decp_data[smallest_vertex1];

//             if (diff>=1e-16 or d>=1e-16){
//                 if (decp_data[false_index]>=decp_data[true_index]){

                    
//                     // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
//                     while(abs(input_data[false_index]-decp_data[false_index] + d)>bound and d>2e-16){
//                                 d/=2;
//                     }
                    
                    
//                     if(decp_data[false_index]>threshold and threshold<decp_data[true_index]){
                            
//                             while(decp_data[false_index] - d < threshold and d>=2e-16)
//                             {
//                                 d/=2;
//                             }
                            
                            
//                     }
//                     else if(threshold>=decp_data[true_index]){
                        
                        
//                         double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/64;
                        
//                         if(diff2>=1e-16){
//                             while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[true_index]){
                                
//                                 diff2/=2;
//                             }
                            
//                             if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
//                                 decp_data[smallest_vertex]-=diff2;
//                                 // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
//                             }
                            
                            
//                         }
                        
//                     }
//                     while(abs(input_data[true_index]-(decp_data[false_index] + diff))>bound and diff>2e-16){
//                                 diff/=2;
//                     }
//                     if(decp_data[true_index]<=threshold and threshold>=decp_data[false_index]){
                            
//                             while(decp_data[false_index] + diff > threshold and diff>=2e-16)
//                             {
//                                 diff/=2;
//                             }
                            
                            
//                     }
//                     // else if(threshold<=decp_data[false_index]){
                        
                        
//                     //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/64;
                        
//                     //     if(diff2>=1e-16){
//                     //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]+diff2<decp_data[false_index]){
                                
//                     //             diff2/=2;
//                     //         }
                            
//                     //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)<=bound){
//                     //             decp_data[smallest_vertex]+=diff2;
//                     //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
//                     //         }
                            
                            
//                     //     }
                        
//                     // }
//                     if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index] and diff>=1e-16){
//                         decp_data[true_index] = decp_data[false_index] + diff;
//                     }
//                     else if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound and d>=1e-16){
//                         decp_data[false_index] -=d;
//                     }
                    
                    
                        
//                 }

//                 else{
//                     de_direction_as[cur] = or_direction_as[cur];
//                 }
                    
//             }
            
//             else{
//                 double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
//                 if (decp_data[false_index]>=decp_data[true_index]){
//                     if(abs(input_data[false_index]-((decp_data[false_index]+input_data[true_index]-bound)/2.0))<=bound){
//                         decp_data[false_index] = (decp_data[false_index]+input_data[true_index]-bound)/2.0;
//                     }
                        
//                     else{
//                         decp_data[false_index] = input_data[false_index] - bound;
//                     }
                    
//                 }
//                 else{
//                     de_direction_as[cur] = or_direction_as[cur];
//                 };        
//             }
            
//         }
//     }

//     else{
//         int cur = index;
        
        
//         while (or_direction_ds[cur] == de_direction_ds[cur]){
            
//             int next_vertex = from_direction_to_index1(cur,de_direction_ds[cur]);
            
//             if(next_vertex == cur){
//                 cur = -1;
//                 break;
//             }
//             if (next_vertex == cur){
//                 cur = next_vertex;
//                 break;
//             }
//             cur = next_vertex;

//             if (cur == -1) break;
                
//         }
    
//         int start_vertex = cur;
        
        
//         if (start_vertex==-1) return 0;
        
//         else{
            
//             int false_index= from_direction_to_index1(cur,de_direction_ds[cur]);
//             int true_index= from_direction_to_index1(cur, or_direction_ds[cur]);
//             if(false_index==true_index) return 0;
            
//             // double diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
//             double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
//             // double d = (bound-abs(input_data[true_index]-decp_data[true_index]))/2.0;
//             double d = (bound+input_data[false_index]-decp_data[false_index])/2.0;
            
//             // diff是对true_index做减法，
//             // d是对false_index做加法
//             // if(decp_data[false_index]>decp_data[true_index]){
                
//             //     de_direction_ds[cur]=or_direction_ds[cur];
            
//             //     return 0;
//             // }
            
              
//             if(diff>=1e-16 or d>=1e-16){
//                 if(decp_data[false_index]<=decp_data[true_index]){
                    
//                     // else{
                        
//                         // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
//                         while(abs(input_data[false_index]-decp_data[false_index] - d)>bound and d>=2e-17){
//                             d/=2;
//                         }
//                         while(abs(input_data[true_index]-(decp_data[false_index] - diff))>bound and diff>=2e-17){
//                                     diff/=2;
//                         }
//                         if(abs(input_data[true_index]-(decp_data[false_index] - diff))<=bound and decp_data[false_index]<=decp_data[true_index]){
//                             // decp_data[false_index] = decp_data[true_index] + diff;
//                             decp_data[true_index] = decp_data[false_index] - diff;
//                         }
//                         if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
//                             decp_data[false_index] += d;
//                         }
                        
                        
                        

//                         // diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
                        
                        
                        
//                         // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
//                         if (decp_data[false_index]==decp_data[true_index]){
//                             if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                        
//                                 decp_data[false_index] += d;
//                         }
                       
//                         }
//                     // }
                    
//                     }
                
//                     else{
//                         de_direction_ds[cur] = or_direction_ds[cur];
//                     }
//             }

//             else{
                
//                 double diff = (bound-(input_data[false_index]-decp_data[false_index]))/100000.0;
                
//                 if(decp_data[false_index]<=decp_data[true_index]){
//                     if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/100000.0))<=bound){
//                         decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/100000.0;
//                     }
//                     else{
//                         decp_data[false_index] = input_data[false_index] + bound;
//                     }
//                     while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound){
//                         diff/=2;
//                     }
//                     if (decp_data[false_index]==decp_data[true_index]){
//                         double diff = (bound-(input_data[false_index]-decp_data[false_index]))/100000.0;
//                         decp_data[false_index]+=diff;
//                     }
                
//                 }
            
//                     else{
//                         de_direction_ds[cur] = or_direction_ds[cur];
//                     }
//             }
//         }
//     }

//     return 0;
// };
int fixpath(int index, int direction, std::atomic<int>* id_array){
    double delta;
    if(direction == 0){

        
        
        int cur = index;
        while (or_direction_as[cur] == de_direction_as[cur]){
            int next_vertex =  from_direction_to_index1(cur,de_direction_as[cur]);
            
            if(de_direction_as[cur]==-1 && next_vertex == cur){
                cur = -1;
                break;
            }
            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        
        if (start_vertex==-1) return 0;
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_as[cur]);
            int true_index= from_direction_to_index1(cur, or_direction_as[cur]);
            if(false_index==true_index) return 0;
            // 对的
            double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // 对的
            double d = (decp_data[false_index]-input_data[false_index]+bound)/2.0;
            // double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            // diff是用来给true_index增加的
            // d是用来给false_index见效的
            // double diff = (input_data[true_index]-bound_data[true_index]-decp_data[false_index])/2.0;
            // double d = (input_data[false_index]-bound_data[false_index]-decp_data[false_index])/2.0;
            // if(wrong_index_as.size()<=50){
            // // pre=1;
            //     cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<endl;
            //     cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<endl;
            //     cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<endl;
            //     cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<endl;
            //     cout<<diff<<endl;
            //     cout<<d<<endl;
            // }
            if(decp_data[false_index]<decp_data[true_index]){
                de_direction_as[cur]=or_direction_as[cur];
            //     if(wrong_maxi_cp.size()==1 and wrong_min_cp.size()==0){
            //     cout<<de_direction_as[64582]<<endl;
            // }
                return 0;
            }
            
            double threshold = -DBL_MAX;;
            int smallest_vertex = false_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[12*false_index+j];
                if(i==-1) continue;
                if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];

            double threshold1 = DBL_MAX;;
            int smallest_vertex1 = true_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[12*true_index+j];
                if(i==-1) continue;
                if(input_data[i]>input_data[true_index] and input_data[i]<threshold1 and i!=true_index){
                    smallest_vertex1 = i;
                    threshold = input_data[i];
                }
            }
            
            threshold1 = decp_data[smallest_vertex1];

            if (diff>=1e-16 or d>=1e-16){
                if (decp_data[false_index]>=decp_data[true_index]){

                    
                    // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                    while(abs(input_data[false_index]-decp_data[false_index] + d)>bound and d>2e-16){
                                d/=2;
                    }
                    
                    
                    // if(decp_data[false_index]>threshold and threshold<decp_data[true_index]){
                            
                    //         while(decp_data[false_index] - d < threshold and d>=2e-16)
                    //         {
                    //             d/=2;
                    //         }
                            
                            
                    // }
                    // else if(threshold>=decp_data[true_index]){
                        
                        
                    //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
                    //     if(diff2>1e-16){
                    //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[true_index]){
                                
                    //             diff2/=2;
                    //         }
                            
                    //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                    //             // decp_data[smallest_vertex]-=diff2;
                    //             delta = -diff2;
                            
                    //             int oldValue = id_array[smallest_vertex];
                    //             // double expected = oldValue;
                                    
                                    
                    //             // if(delta>10000){
                    //             //         printf("chuchuo");
                    //             //     }
                    //             // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                    //             if (cur > oldValue) {
                    //                     atomicCAS(&id_array[smallest_vertex], id_array[smallest_vertex], cur);
                    //                     d_deltaBuffer[smallest_vertex] = delta;
                    //                     // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                    //                 }
                    //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                    //         }
                            
                            
                    //     }
                        
                    // }
                    while(abs(input_data[true_index]-(decp_data[false_index] + diff))>bound and diff>2e-16){
                                diff/=2;
                    }
                    if(decp_data[true_index]<=threshold and threshold>=decp_data[false_index]){
                            
                            while(decp_data[false_index] + diff > threshold and diff>=2e-16)
                            {
                                diff/=2;
                            }
                            
                            
                    }
                    // else if(threshold<=decp_data[false_index]){
                        
                        
                    //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
                    //     if(diff2>1e-16){
                    //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]+diff2<decp_data[false_index]){
                                
                    //             diff2/=2;
                    //         }
                            
                    //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)<=bound){
                    //             decp_data[smallest_vertex]+=diff2;
                    //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                    //         }
                            
                            
                    //     }
                        
                    // }
                    if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index]){
                        // decp_data[true_index] = decp_data[false_index] + diff;
                        delta = decp_data[false_index] + diff - decp_data[true_index];
                            
                        // int oldValue = id_array[true_index];
                        // // double expected = oldValue;
                            
                            
                        // // if(delta>10000){
                        // //         printf("chuchuo");
                        // //     }
                        // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        // if (cur > oldValue) {
                        //         atomicCAS(&id_array[true_index], id_array[true_index], cur);
                        //         d_deltaBuffer[true_index] = delta;
                        //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                        //     }
                        // d_stacks[true_index].push(delta);
                        
                        double oldValue = d_deltaBuffer[true_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(true_index);
                        if (delta > oldValue) {
                                swap(true_index, delta);
                                
                            }
                    }
                    if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
                        
                        // decp_data[false_index] -=d;
                        delta = -d;
                        // int oldValue = id_array[false_index];
                        // // double expected = oldValue;
                            
                            
                        // // if(delta>10000){
                        // //         printf("chuchuo");
                        // //     }
                        // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        // if (cur > oldValue) {
                        //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                        //         d_deltaBuffer[false_index] = delta;
                        //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                        //     }
                        
                        
                        double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                    }
                    
                    
                        
                }

                else{
                    de_direction_as[cur] = or_direction_as[cur];
                }
                    
            }
            
            else{
                //对的
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                // if(wrong_index_as.size()==2){
                //     cout<<diff<<endl;
                //     cout<<false_index<<endl;
                // }
                if (decp_data[false_index]>=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[false_index]+input_data[true_index]-bound)/2.0))<=bound){
                        // decp_data[false_index] = (decp_data[false_index]+input_data[true_index]-bound)/2.0;
                        delta = (decp_data[false_index]+input_data[true_index]-bound)/2.0-decp_data[false_index];
                        // int oldValue = id_array[false_index];
                        // // double expected = oldValue;
                            
                            
                        // // if(delta>10000){
                        // //         printf("chuchuo");
                        // //     }
                        // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        // if (cur > oldValue) {
                        //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                        //         d_deltaBuffer[false_index] = delta;
                        //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                        //     }
                        // d_stacks[false_index].push(delta);
                        
                        double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                    }
                        
                    else{
                        // decp_data[false_index] = input_data[false_index] - bound;
                        delta =  input_data[false_index] - bound-decp_data[false_index];
                        // int oldValue = id_array[false_index];
                        // // double expected = oldValue;
                            
                            
                        // // if(delta>10000){
                        // //         printf("chuchuo");
                        // //     }
                        // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        // if (cur > oldValue) {
                        //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                        //         d_deltaBuffer[false_index] = delta;
                        //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                        //     }
                        // d_stacks[false_index].push(delta);
                        
                        double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                    }
                    
                }
                else{
                    de_direction_as[cur] = or_direction_as[cur];
                };        
            }
            
        }
        
    }

    else 
    {
        
            
        
        int cur = index;
        
        
        while (or_direction_ds[cur] == de_direction_ds[cur]){
            
            int next_vertex = from_direction_to_index1(cur,de_direction_ds[cur]);
            
            // if(de_direction_ds[cur]==-1 && next_vertex == cur){
            //     if(wrong_index_ds.size()==4){
            //         cout<<cur<<", "<<index <<", "<<de_direction_ds[cur]<<", "<<or_direction_ds[cur]<<endl;
            //     }
            //     cur = -1;
            //     break;
            // }
            if (next_vertex == cur){
                cur = next_vertex;
                break;
            }
            cur = next_vertex;

            // if (cur == -1) break;
                
        }
    
        int start_vertex = cur;
        // if(wrong_index_ds.size()==4){
        //     cout<<"修复的时候变成了:" <<endl;
        //     cout<<start_vertex<<", "<<de_direction_ds[start_vertex]<<", "<<or_direction_ds[start_vertex]<<endl;
        // }
        if (start_vertex==-1) return 0;
        
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_ds[cur]);
            int true_index= from_direction_to_index1(cur, or_direction_ds[cur]);
            if(false_index==true_index) return 0;

            // double diff = (input_data[true_index]+bound-decp_data[false_index])/2.0;
            double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // double d = (input_data[false_index]bound-decp_data[false_index])/1000.0;
            // double d = (input_data[false_index]+bound-decp_data[false_index])/2.0;
            // double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            // // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // double d = (input_data[false_index]-bound-decp_data[false_index])/2.0;
            double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            // diff是用来给true_index增加的
            // d是用来给false_index见效的
            // if(wrong_index_as.size()<=10){
            //     cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<endl;
            //     cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<endl;
            //     cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<endl;   
            //     cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<endl;                     
            // }
            if(decp_data[false_index]>decp_data[true_index]){
                de_direction_ds[cur]=or_direction_ds[cur];
                return 0;
            }
            
            if(diff>=1e-16 or d>=1e-16){
                if(decp_data[false_index]<=decp_data[true_index]){
                    
                    // else{
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        while(abs(input_data[false_index]-decp_data[false_index] - d)>bound and d>=2e-17){
                            d/=2;
                        }
                        while(abs(input_data[true_index]-(decp_data[false_index] - diff))>bound and diff>=2e-17){
                                    diff/=2;
                        }
                        if(abs(input_data[true_index]-(decp_data[false_index] - diff))<=bound and decp_data[false_index]<=decp_data[true_index]){
                            // decp_data[false_index] = decp_data[true_index] + diff;
                            // decp_data[true_index] = decp_data[false_index] - diff;
                            delta =  decp_data[false_index] - diff- decp_data[true_index];
                            // int oldValue = id_array[true_index];
                            // // double expected = oldValue;
                                
                                
                            // // if(delta>10000){
                            // //         printf("chuchuo");
                            // //     }
                            // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            // if (cur > oldValue) {
                            //         atomicCAS(&id_array[true_index], id_array[true_index], cur);
                            //         d_deltaBuffer[true_index] = delta;
                            //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            //     }
                            // d_stacks[true_index].push(delta);
                            
                            double oldValue = d_deltaBuffer[true_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(true_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(true_index, delta);
                                
                            }
                        }
                        if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                            
                            delta =  d;
                            // int oldValue = id_array[false_index];
                            // // double expected = oldValue;
                                
                                
                            // // if(delta>10000){
                            // //         printf("chuchuo");
                            // //     }
                            // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            // if (cur > oldValue) {
                            //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                            //         d_deltaBuffer[false_index] = delta;
                            //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            //     }
                            // d_stacks[false_index].push(delta);
                            
                            double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                        }
                        
                        
                        

                        // diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
                        
                        
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        if (decp_data[false_index]==decp_data[true_index]){
                            if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                        
                                // decp_data[false_index] += d;
                                delta =  d;
                                // int oldValue = id_array[false_index];
                                // // double expected = oldValue;
                                    
                                    
                                // // if(delta>10000){
                                // //         printf("chuchuo");
                                // //     }
                                // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                                // if (cur > oldValue) {
                                //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                                //         d_deltaBuffer[false_index] = delta;
                                //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                //     }
                                // d_stacks[false_index].push(delta);
                                
                                double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                        }
                       
                    }
                    // }
                    
                }
            
                else{
                    de_direction_ds[cur] = or_direction_ds[cur];
                }
            }

            else{
                
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                
                if(decp_data[false_index]<=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/2.0))<=bound){
                        // decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/2.0;
                        delta =  (decp_data[true_index]+input_data[true_index]+bound)/2.0 - decp_data[false_index];
                            // int oldValue = id_array[false_index];
                            // // double expected = oldValue;
                                
                                
                            // // if(delta>10000){
                            // //         printf("chuchuo");
                            // //     }
                            // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            // if (cur > oldValue) {
                            //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                            //         d_deltaBuffer[false_index] = delta;
                            //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            //     }
                            // d_stacks[false_index].push(delta);
                            
                            double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                    }
                    else{
                        // decp_data[false_index] = input_data[false_index] + bound;
                        delta =  input_data[false_index] + bound - decp_data[false_index];
                            // int oldValue = id_array[false_index];
                            // // double expected = oldValue;
                                
                                
                            // // if(delta>10000){
                            // //         printf("chuchuo");
                            // //     }
                            // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            // if (cur > oldValue) {
                            //         atomicCAS(&id_array[false_index], id_array[false_index],cur);
                            //         d_deltaBuffer[false_index] = delta;
                            //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            //     }
                            // d_stacks[false_index].push(delta);
                            
                            double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                    }
                    while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound and diff>=2e-17){
                        diff/=2;
                    }
                    if (decp_data[false_index]==decp_data[true_index]){
                            double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                            // decp_data[false_index]+=diff;
                            delta =  diff;
                            // int oldValue = id_array[false_index];
                            // // double expected = oldValue;
                                
                                
                            // // if(delta>10000){
                            // //         printf("chuchuo");
                            // //     }
                            // // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            // if (cur > oldValue) {
                            //         atomicCAS(&id_array[false_index], id_array[false_index], cur);
                            //         d_deltaBuffer[false_index] = delta;
                            //         // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            //     }
                            // d_stacks[false_index].push(delta);
                            
                            double oldValue = d_deltaBuffer[false_index];
                        // d_stacks.push(delta);
                        // id_stacks.push(false_index);
                        if (delta > oldValue) {
                                // atomicCAS(&id_array[index], id_array[index], index);
                                // d_deltaBuffer[index] = -d;
                                swap(false_index, delta);
                                
                            }
                    }
                
                }
            
                else{
                    de_direction_ds[cur] = or_direction_ds[cur];
                }
            }
        }
    
    }
    return 0;
}
int fixpath11(int index, int direction, std::atomic<int>* id_array){
    
    double delta;
    if(direction == 0){
        if(lowGradientIndices[index]==0){

        
        // int index = all_p_max[index];
        int cur = index;
        while (or_direction_as[cur] == de_direction_as[cur]){
            int next_vertex =  from_direction_to_index1(cur,de_direction_as[cur]);
            
            if(de_direction_as[cur]==-1 && next_vertex == cur){
                cur = -1;
                break;
            }
            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        
        if (start_vertex==-1) return 0;
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_as[cur]);
            int true_index= from_direction_to_index1(cur, or_direction_as[cur]);
            if(false_index==true_index) return 0;
            // 对的
            double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // 对的
            double d = (decp_data[false_index]-input_data[false_index]+bound)/2.0;
            // double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            // diff是用来给true_index增加的
            // d是用来给false_index见效的
            // double diff = (input_data[true_index]-bound_data[true_index]-decp_data[false_index])/2.0;
            // double d = (input_data[false_index]-bound_data[false_index]-decp_data[false_index])/2.0;
            // if(wrong_index_as.size()<=50){
            // // pre=1;
            //     cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<endl;
            //     cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<endl;
            //     cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<endl;
            //     cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<endl;
            //     cout<<diff<<endl;
            //     cout<<d<<endl;
            // }
            if(decp_data[false_index]<decp_data[true_index]){
                de_direction_as[cur]=or_direction_as[cur];
            //     if(wrong_maxi_cp.size()==1 and wrong_min_cp.size()==0){
            //     cout<<de_direction_as[64582]<<endl;
            // }
                return 0;
            }
            
            double threshold = std::numeric_limits<double>::lowest();
            int smallest_vertex = false_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[12*false_index+j];
                if(i==-1) continue;
                if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];

            double threshold1 = std::numeric_limits<double>::max();
            int smallest_vertex1 = true_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[12*true_index+j];
                if(i==-1) continue;
                if(input_data[i]>input_data[true_index] and input_data[i]<threshold1 and i!=true_index){
                    smallest_vertex1 = i;
                    threshold = input_data[i];
                }
            }
            
            threshold1 = decp_data[smallest_vertex1];

            if (diff>=1e-16 or d>=1e-16){
                if (decp_data[false_index]>=decp_data[true_index]){

                    
                    // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                    while(abs(input_data[false_index]-decp_data[false_index] + d)>bound and d>2e-16){
                                d/=2;
                    }
                    
                    
                    if(decp_data[false_index]>threshold and threshold<decp_data[true_index]){
                            
                            while(decp_data[false_index] - d < threshold and d>=2e-16)
                            {
                                d/=2;
                            }
                            
                            
                    }
                    else if(threshold>=decp_data[true_index]){
                        
                        
                        double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
                        if(diff2>1e-16){
                            while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[true_index]){
                                
                                diff2/=2;
                            }
                            
                            // if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                            //     // decp_data[smallest_vertex]-=diff2;
                            //     delta = -diff2;
                            
                            //     double oldValue = id_array[smallest_vertex];
                            //     // double expected = oldValue;
                                    
                                    
                            //     // if(delta>10000){
                            //     //         printf("chuchuo");
                            //     //     }
                            //     // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            //     if (delta > oldValue) {
                            //             swap(index,delta);
                            //             // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            //         }
                            //     // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                            // }
                            
                            
                        }
                        
                    }
                    while(abs(input_data[true_index]-(decp_data[false_index] + diff))>bound and diff>2e-16){
                                diff/=2;
                    }
                    if(decp_data[true_index]<=threshold and threshold>=decp_data[false_index]){
                            
                            while(decp_data[false_index] + diff > threshold and diff>=2e-16)
                            {
                                diff/=2;
                            }
                            
                            
                    }
                    // else if(threshold<=decp_data[false_index]){
                        
                        
                    //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
                    //     if(diff2>1e-16){
                    //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]+diff2<decp_data[false_index]){
                                
                    //             diff2/=2;
                    //         }
                            
                    //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)<=bound){
                    //             decp_data[smallest_vertex]+=diff2;
                    //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                    //         }
                            
                            
                    //     }
                        
                    // }
                    if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index]){
                        // decp_data[true_index] = decp_data[false_index] + diff;
                        delta = decp_data[false_index] + diff-decp_data[true_index];
                            
                        double oldValue = d_deltaBuffer[true_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(true_index, delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                    if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
                        
                        // decp_data[false_index] -=d;
                        delta = -d;
                        double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index, delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                    
                    
                        
                }

                else{
                    de_direction_as[cur] = or_direction_as[cur];
                }
                    
            }
            
            else{
                //对的
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                // if(wrong_index_as.size()==2){
                //     cout<<diff<<endl;
                //     cout<<false_index<<endl;
                // }
                if (decp_data[false_index]>=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[false_index]+input_data[true_index]-bound)/2.0))<=bound){
                        // decp_data[false_index] = (decp_data[false_index]+input_data[true_index]-bound)/2.0;
                        delta = (decp_data[false_index]+input_data[true_index]-bound)/2.0-decp_data[false_index];
                        double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                        
                    else{
                        // decp_data[false_index] = input_data[false_index] - bound;
                        delta =  input_data[false_index] - bound-decp_data[false_index];
                        double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                    
                }
                else{
                    de_direction_as[cur] = or_direction_as[cur];
                };        
            }
            
        }
        }
    }

    else 
    {
        if(lowGradientIndices[index]==0){
        
        int cur = index;
        
        
        while (or_direction_ds[cur] == de_direction_ds[cur]){
            
            int next_vertex = from_direction_to_index1(cur,de_direction_ds[cur]);
            
            // if(de_direction_ds[cur]==-1 && next_vertex == cur){
            //     if(wrong_index_ds.size()==4){
            //         cout<<cur<<", "<<index <<", "<<de_direction_ds[cur]<<", "<<or_direction_ds[cur]<<endl;
            //     }
            //     cur = -1;
            //     break;
            // }
            if (next_vertex == cur){
                cur = next_vertex;
                break;
            }
            cur = next_vertex;

            // if (cur == -1) break;
                
        }
    
        int start_vertex = cur;
        // if(wrong_index_ds.size()==4){
        //     cout<<"修复的时候变成了:" <<endl;
        //     cout<<start_vertex<<", "<<de_direction_ds[start_vertex]<<", "<<or_direction_ds[start_vertex]<<endl;
        // }
        if (start_vertex==-1) return 0;
        
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_ds[cur]);
            int true_index= from_direction_to_index1(cur, or_direction_ds[cur]);
            if(false_index==true_index) return 0;

            // double diff = (input_data[true_index]+bound-decp_data[false_index])/2.0;
            double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // double d = (input_data[false_index]bound-decp_data[false_index])/1000.0;
            // double d = (input_data[false_index]+bound-decp_data[false_index])/2.0;
            // double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            // // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // double d = (input_data[false_index]-bound-decp_data[false_index])/2.0;
            double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            // diff是用来给true_index增加的
            // d是用来给false_index见效的
            // if(wrong_index_as.size()<=10){
            //     cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<endl;
            //     cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<endl;
            //     cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<endl;   
            //     cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<endl;                     
            // }
            if(decp_data[false_index]>decp_data[true_index]){
                de_direction_ds[cur]=or_direction_ds[cur];
                return 0;
            }
            
            if(diff>=1e-16 or d>=1e-16){
                if(decp_data[false_index]<=decp_data[true_index]){
                    
                    // else{
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        while(abs(input_data[false_index]-decp_data[false_index] - d)>bound and d>=2e-17){
                            d/=2;
                        }
                        while(abs(input_data[true_index]-(decp_data[false_index] - diff))>bound and diff>=2e-17){
                                    diff/=2;
                        }
                        if(abs(input_data[true_index]-(decp_data[false_index] - diff))<=bound and decp_data[false_index]<=decp_data[true_index]){
                            // decp_data[false_index] = decp_data[true_index] + diff;
                            // decp_data[true_index] = decp_data[false_index] - diff;
                            delta =  decp_data[false_index] - diff- decp_data[true_index];
                            double oldValue = d_deltaBuffer[true_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(true_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                        }
                        if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                            
                            delta =  d;
                            double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                        }
                        
                        
                        

                        // diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
                        
                        
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        if (decp_data[false_index]==decp_data[true_index]){
                            if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                        
                                // decp_data[false_index] += d;
                                delta =  d;
                                double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                        }
                       
                    }
                    // }
                    
                }
            
                else{
                    de_direction_ds[cur] = or_direction_ds[cur];
                }
            }

            else{
                
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                
                if(decp_data[false_index]<=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/2.0))<=bound){
                        // decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/2.0;
                        delta =  (decp_data[true_index]+input_data[true_index]+bound)/2.0 - decp_data[false_index];
                            double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                    else{
                        // decp_data[false_index] = input_data[false_index] + bound;
                        delta =  input_data[false_index] + bound - decp_data[false_index];
                            double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                    while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound and diff>=2e-17){
                        diff/=2;
                    }
                    if (decp_data[false_index]==decp_data[true_index]){
                        double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                        // decp_data[false_index]+=diff;
                        delta =  diff;
                            double oldValue = d_deltaBuffer[false_index];
                                // double expected = oldValue;
                                    
                                    
                        // if(delta>10000){
                        //         printf("chuchuo");
                        //     }
                        // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                swap(false_index,delta);
                                // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                            }
                    }
                
                }
            
                else{
                    de_direction_ds[cur] = or_direction_ds[cur];
                }
            }
        }
    }
    }
    return 0 ;
    }
int fixpath1(int index, int direction, std::atomic<int>* id_array){
    double delta;
    if(direction==0){
        int cur = index;
        while (or_direction_as[cur] == de_direction_as[cur]){
            int next_vertex =  from_direction_to_index1(cur,de_direction_as[cur]);
            
            if(de_direction_as[cur]==-1 && next_vertex == cur){
                cur = -1;
                break;
            }
            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        
        if (start_vertex==-1) return 0;
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_as[cur]);
            int true_index= from_direction_to_index1(cur, or_direction_as[cur]);
            if(false_index==true_index) return 0;
            // 对的
            double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // 对的
            double d = (decp_data[false_index]-input_data[false_index]+bound)/2.0;
            // double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            // diff是用来给true_index增加的
            // d是用来给false_index见效的
            // double diff = (input_data[true_index]-bound_data[true_index]-decp_data[false_index])/2.0;
            // double d = (input_data[false_index]-bound_data[false_index]-decp_data[false_index])/2.0;
            // if(wrong_index_as.size()<=50){
            // // pre=1;
            //     std::cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<std::endl;
            //     std::cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<std::endl;
            //     std::cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<std::endl;
            //     std::cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<std::endl;
            //     std::cout<<diff<<std::endl;
            //     std::cout<<d<<std::endl;
            // }
            if(decp_data[false_index]<decp_data[true_index]){
                de_direction_as[cur]=or_direction_as[cur];
            //     if(wrong_maxi_cp.size()==1 and wrong_min_cp.size()==0){
            //     std::cout<<de_direction_as[64582]<<std::endl;
            // }
                return 0;
            }
            double threshold = std::numeric_limits<double>::lowest();
            int smallest_vertex = false_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[index*12+j];
                if(i==-1) continue;
                if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];

            double threshold1 = std::numeric_limits<double>::max();
            int smallest_vertex1 = true_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[index*12+j];
                if(i==-1) continue;
                if(input_data[i]>input_data[true_index] and input_data[i]<threshold1 and i!=true_index){
                    smallest_vertex1 = i;
                    threshold = input_data[i];
                }
            }
            
            threshold1 = decp_data[smallest_vertex1];

            if (diff>=1e-16 or d>=1e-16){
                if (decp_data[false_index]>=decp_data[true_index]){

                    
                    // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                    while(abs(input_data[false_index]-decp_data[false_index] + d)>bound and d>2e-16){
                                d/=2;
                    }
                    
                    
                    if(decp_data[false_index]>threshold and threshold<decp_data[true_index]){
                            
                            while(decp_data[false_index] - d < threshold and d>=2e-16)
                            {
                                d/=2;
                            }
                            
                            
                    }
                    else if(threshold>=decp_data[true_index]){
                        
                        
                        double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
                        if(diff2>1e-16){
                            while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[true_index]){
                                
                                diff2/=2;
                            }
                            
                            // if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                            //     delta = -diff2;
                            //     double oldValue = id_array[smallest_vertex];
                            //         // int expected = d_deltaBuffer[index];
                                
                            
                            //     // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            //     // if (delta > oldValue) {
                            //     //         // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                        
                            //     //         swap(false_index,delta);
                                        
                            //     //             // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                        
                            //     //     }
                            //     // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
                            // }
                            
                            
                        }
                        
                    }
                    while(abs(input_data[true_index]-(decp_data[false_index] + diff))>bound and diff>2e-16){
                                diff/=2;
                    }
                    if(decp_data[true_index]<=threshold and threshold>=decp_data[false_index]){
                            
                            while(decp_data[false_index] + diff > threshold and diff>=2e-16)
                            {
                                diff/=2;
                            }
                            
                            
                    }
                    // else if(threshold<=decp_data[false_index]){
                        
                        
                    //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
                    //     if(diff2>1e-16){
                    //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]+diff2<decp_data[false_index]){
                                
                    //             diff2/=2;
                    //         }
                            
                    //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)<=bound){
                    //             decp_data[smallest_vertex]+=diff2;
                    //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
                    //         }
                            
                            
                    //     }
                        
                    // }
                    if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index]){
                        delta = decp_data[false_index] + diff-decp_data[true_index];
                        double oldValue = d_deltaBuffer[true_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(true_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                    if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
                        
                        // decp_data[false_index] -=d;
                        delta = -d;
                        double oldValue = d_deltaBuffer[false_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(false_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                    
                    
                        
                }

                else{
                    de_direction_as[cur] = or_direction_as[cur];
                }
                    
            }
            
            else{
                //对的
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                // if(wrong_index_as.size()==2){
                //     std::cout<<diff<<std::endl;
                //     std::cout<<false_index<<std::endl;
                // }
                if (decp_data[false_index]>=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[false_index]+input_data[true_index]-bound)/2.0))<=bound){
                        // decp_data[false_index] = (decp_data[false_index]+input_data[true_index]-bound)/2.0;
                        delta = (decp_data[false_index]+input_data[true_index]-bound)/2.0 - decp_data[false_index];
                        double oldValue = d_deltaBuffer[false_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(false_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                        
                    else{
                        // decp_data[false_index] = input_data[false_index] - bound;
                        delta = input_data[false_index] - bound - decp_data[false_index];
                        double oldValue = d_deltaBuffer[false_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(false_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                    
                }
                else{
                    de_direction_as[cur] = or_direction_as[cur];
                };        
            }
            
        }
    }

    else{
        int cur = index;
        
        
        while (or_direction_ds[cur] == de_direction_ds[cur]){
            
            int next_vertex = from_direction_to_index1(cur,de_direction_ds[cur]);
            
            // if(de_direction_ds[cur]==-1 && next_vertex == cur){
            //     if(wrong_index_ds.size()==4){
            //         std::cout<<cur<<", "<<index <<", "<<de_direction_ds[cur]<<", "<<or_direction_ds[cur]<<std::endl;
            //     }
            //     cur = -1;
            //     break;
            // }
            if (next_vertex == cur){
                cur = next_vertex;
                break;
            }
            cur = next_vertex;

            // if (cur == -1) break;
                
        }
    
        int start_vertex = cur;
        // if(wrong_index_ds.size()==4){
        //     std::cout<<"修复的时候变成了:" <<std::endl;
        //     std::cout<<start_vertex<<", "<<de_direction_ds[start_vertex]<<", "<<or_direction_ds[start_vertex]<<std::endl;
        // }
        if (start_vertex==-1) return 0;
        
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_ds[cur]);
            int true_index= from_direction_to_index1(cur, or_direction_ds[cur]);
            if(false_index==true_index) return 0;

            // double diff = (input_data[true_index]+bound-decp_data[false_index])/2.0;
            double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // double d = (input_data[false_index]bound-decp_data[false_index])/1000.0;
            // double d = (input_data[false_index]+bound-decp_data[false_index])/2.0;
            // double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            // // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            // double d = (input_data[false_index]-bound-decp_data[false_index])/2.0;
            double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            // diff是用来给true_index增加的
            // d是用来给false_index见效的
            // if(wrong_index_as.size()<=10){
            //     std::cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<std::endl;
            //     std::cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<std::endl;
            //     std::cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<std::endl;   
            //     std::cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<std::endl;                     
            // }
            if(decp_data[false_index]>decp_data[true_index]){
                de_direction_ds[cur]=or_direction_ds[cur];
                return 0;
            }
            
            if(diff>=1e-16 or d>=1e-16){
                if(decp_data[false_index]<=decp_data[true_index]){
                    
                    // else{
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        while(abs(input_data[false_index]-decp_data[false_index] - d)>bound and d>=2e-17){
                            d/=2;
                        }
                        while(abs(input_data[true_index]-(decp_data[false_index] - diff))>bound and diff>=2e-17){
                                    diff/=2;
                        }
                        if(abs(input_data[true_index]-(decp_data[false_index] - diff))<=bound and decp_data[false_index]<=decp_data[true_index]){
                            // decp_data[false_index] = decp_data[true_index] + diff;
                            // decp_data[true_index] = decp_data[false_index] - diff;
                            delta = decp_data[false_index] - diff - decp_data[true_index];
                            double oldValue = d_deltaBuffer[true_index];
                                        // int expected = d_deltaBuffer[index];
                                    
                                
                                    // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                    
                                    swap(true_index,delta);
                                    
                                        // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                    
                                }
                        }
                        if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                            // decp_data[false_index] += d;
                            delta = d;
                            double oldValue = d_deltaBuffer[false_index];
                                        // int expected = d_deltaBuffer[index];
                                    
                                
                                    // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                    
                                    swap(false_index,delta);
                                    
                                        // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                    
                                }
                        }
                        
                        
                        

                        // diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
                        
                        
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        if (decp_data[false_index]==decp_data[true_index]){
                            if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                        
                                delta = d;
                            double oldValue = d_deltaBuffer[false_index];
                                        // int expected = d_deltaBuffer[index];
                                    
                                
                                    // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                            if (delta > oldValue) {
                                    // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                    
                                    swap(false_index,delta);
                                        // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                    
                                }
                        }
                       
                    }
                    // }
                    
                }
            
                else{
                    de_direction_ds[cur] = or_direction_ds[cur];
                }
            }

            else{
                
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                
                if(decp_data[false_index]<=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/2.0))<=bound){
                        // decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/2.0;
                        delta = (decp_data[true_index]+input_data[true_index]+bound)/2.0 - decp_data[false_index];
                        double oldValue = d_deltaBuffer[false_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(false_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                    else{
                        // decp_data[false_index] = input_data[false_index] + bound;
                        delta = input_data[false_index] + bound - decp_data[false_index];
                        double oldValue = d_deltaBuffer[false_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(false_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                    while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound and diff>=2e-17){
                        diff/=2;
                    }
                    if (decp_data[false_index]==decp_data[true_index]){
                        double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                        // decp_data[false_index]+=diff;
                        delta = diff;
                        double oldValue = d_deltaBuffer[false_index];
                                    // int expected = d_deltaBuffer[index];
                                
                            
                                // oldValue = atomicMaxDouble(&d_deltaBuffer[index], delta);
                        if (delta > oldValue) {
                                // printf("改变钱: %d\n",d_deltaBuffer[index]);
                                
                                swap(false_index,delta);
                                
                                    // if(delta>1000) printf("delta出错： %.17lf %d\n", delta, index);
                                
                            }
                    }
                
                }
            
                else{
                    de_direction_ds[cur] = or_direction_ds[cur];
                }
            }
        }
    }

    return 0;
};
// int get_wrong_index_cp(){
    
//     return 0;
// };
double calculateMSE(const std::vector<double>& original, const std::vector<double>& compressed) {
    if (original.size() != compressed.size()) {
        throw std::invalid_argument("The size of the two vectors must be the same.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < original.size(); i++) {
        mse += std::pow(static_cast<double>(original[i]) - compressed[i], 2);
    }
    mse /= original.size();
    return mse;
}

double calculatePSNR(const std::vector<double>& original, const std::vector<double>& compressed, double maxValue) {
    double mse = calculateMSE(original, compressed);
    if (mse == 0) {
        return std::numeric_limits<double>::infinity(); // Perfect match
    }
    double psnr = -20.0*log10(sqrt(mse)/maxValue);
    return psnr;
}

double get_wrong_index_path(){
    
    wrong_index_as.clear();
    wrong_index_ds.clear();
    double cnt = 0.0;
    for (int index = 0; index < size2; ++index) {
        if(lowGradientIndices[index] == 1){
            continue;
        }
        if(or_label[index*2+1] != dec_label[index*2+1] || or_label[index*2] != dec_label[index*2]){
            cnt+=1.0;
        }
        if (or_label[index*2+1] != dec_label[index*2+1]) {
            wrong_index_as.push_back(index);
        }
        if (or_label[index*2] != dec_label[index*2]) {
            wrong_index_ds.push_back(index);
        }
}
    // std::cout<<cnt/size2<<std::endl;
    double result = static_cast<double>(cnt) / static_cast<double>(size2);
    return result;
};
// bool compareIndices(int i1, int i2, const std::vector<double>& data) {
//     if (data[i1] == data[i2]) {
//         return i1<i2; // 注意这里是‘>’
//     }
    
//     return data[i1] < data[i2];
// }

void get_false_criticle_points(){
    count_max=0;
    count_min=0;

    count_f_max=0;
    count_f_min=0;

    #pragma omp parallel for
    for (auto i = 0; i < size2; i ++) {
            if(lowGradientIndices[i] == 1){
                continue;
            }
            bool is_maxima = true;
            bool is_minima = true;
        
            for (int index=0;index<12;index++) {
                int j = adjacency[i*12+index];
                if(j==-1){
                    break;
                }
                if(lowGradientIndices[j]==1){
                    continue;
                }
            
                
            if (decp_data[j] > decp_data[i]) {
                
                is_maxima = false;
                
                break;
            }
            else if(decp_data[j] == decp_data[i] and j>i){
                is_maxima = false;
                break;
            }
        }
        for (int index=0;index< 12;index++) {
            int j = adjacency[i*12+index];
            if(j==-1){
                break;
            }
            if(lowGradientIndices[j]==1){
                    continue;
            }
            
            if (decp_data[j] < decp_data[i]) {
                is_minima = false;
                break;
            }
            else if(decp_data[j] == decp_data[i] and j<i){
                is_minima = false;
                break;
            }
        }
        
        
        if((is_maxima && or_direction_as[i]!=-1) or (!is_maxima && or_direction_as[i]==-1)){
            int idx_fp_max = atomic_fetch_add_explicit(&count_f_max, 1, memory_order_relaxed);
            // if(i==6345199){
            //     printf("%d %d \n",is_maxima,or_direction_as[i]);
            // }
            all_max[idx_fp_max] = i;
            
        }
        
        else if ((is_minima && or_direction_ds[i]!=-1) or (!is_minima && or_direction_ds[i]==-1)) {
            int idx_fp_min = atomic_fetch_add_explicit(&count_f_min, 1, memory_order_relaxed);// in one instruction
            
            all_min[idx_fp_min] = i;
            
        }    
        
}
    

}
// extern "C" {
//     int mainCuda(std::vector<int> a, std::vector<int> b);
// }
// fix_process
// extern __global__ void init_or_data1(int numElements);
// extern void init_or_data(std::vector<int> *a, std::vector<int> *b,std::vector<int> *c, std::vector<int> *d, std::vector<double> *input_data,std::vector<double> *decp_data, int num);
// extern int fix_process(std::vector<double>& decp_data);

// void getlabel(int i){
    
    
    
//     int cur = dec_label[i*2+1];
//     int next_vertex;
//     // std::cout<<cur<<std::endl;
//     if (cur==-1){
        
//         return;
//     }
//     else if (de_direction_as[cur]!=-1){
        
//         // std::cout<<cur<<" "<<de_direction_as[cur]<<std::endl;
//         int direc = de_direction_as[cur];
//         int row = cur/width;
//         int rank1 = cur%width;
        
//         switch (direc) {
//             case 1:
//                 next_vertex = (row)*width + (rank1-1);
//                 break;
//             case 2:
//                 next_vertex = (row-1)*width + (rank1);
//                 break;
//             case 3:
//                 next_vertex = (row-1)*width + (rank1+1);
//                 break;
//             case 4:
//                 next_vertex = (row)*width + (rank1+1);
//                 break;
//             case 5:
//                 next_vertex = (row+1)*width + (rank1);
//                 break;
//             case 6:
//                 next_vertex = (row+1)*width + (rank1-1);
//                 break;
//         };

//         cur = next_vertex;
        
//         if (de_direction_as[cur] != -1){
            
//             un_sign_as+=1;
//         }

//         if(de_direction_as[i]!=-1){
//             dec_label[i*2+1] = cur;
            
//         }
//         else{
//             dec_label[i*2+1] = -1;
//         };
        
//     }

    
    
//     cur = dec_label[i*2];
//     int next_vertex1;
//     if(cur==-1){
//         return;
//     }
//     if (de_direction_as[cur]!=-1){
//         // printf("%d\n", cur);
//         int direc = de_direction_ds[cur];
//         int row = cur/width;
//         int rank1 = cur%width;
        
//         switch (direc) {
//             case 1:
//                 next_vertex1 = (row)*width + (rank1-1);
//                 break;
//             case 2:
//                 next_vertex1 = (row-1)*width + (rank1);
//                 break;
//             case 3:
//                 next_vertex1 = (row-1)*width + (rank1+1);
//                 break;
//             case 4:
//                 next_vertex1 = (row)*width + (rank1+1);
//                 break;
//             case 5:
//                 next_vertex1 = (row+1)*width + (rank1);
//                 break;
//             case 6:
//                 next_vertex1 = (row+1)*width + (rank1-1);
//                 break;
//         };

//         cur = next_vertex1;
        
//         if (de_direction_ds[cur] != -1){
//             un_sign_ds+=1;
//             // printf("%d \n",i);
//         }

//         if(de_direction_ds[i]!=-1){
//             dec_label[i*2] = cur;
            
//         }
//         else{
//             dec_label[i*2] = -1;
//         };
        
//     }

    

// }
double maxAbsoluteDifference(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors are of unequal size." << std::endl;
        return -1; // Or handle the error as per your need
    }

    double maxDiff = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = std::abs(vec1[i] - vec2[i]);
        if (diff < maxDiff) {
            maxDiff = diff;
        }
    }

    return maxDiff;
}
int main(int argc, char** argv){
    
    std::string dimension = argv[1];
    double range = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int mode = std::stoi(argv[4]);
    int number_of_thread = std::stoi(argv[5]);
    omp_set_num_threads(number_of_thread);
    
    double target_br;
    if(mode==1){
        target_br = std::stod(argv[5]);
    }
    std::istringstream iss(dimension);
    char delimiter;
    std::string filename;
    if (std::getline(iss, filename, ',')) {
        if (iss >> width >> delimiter && delimiter == ',' &&
            iss >> height >> delimiter && delimiter == ',' &&
            iss >> depth) {
            std::cout << "Filename: " << filename << std::endl;
            std::cout << "Width: " << width << std::endl;
            std::cout << "Height: " << height << std::endl;
            std::cout << "Depth: " << depth << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for filename" << std::endl;
    }

    
    inputfilename = "experiment_data/"+filename+".bin";
    auto start = std::chrono::high_resolution_clock::now();
    input_data = getdata2(inputfilename);
    auto min_it = std::min_element(input_data.begin(), input_data.end());
    auto max_it = std::max_element(input_data.begin(), input_data.end());
    double minValue = *min_it;
    double maxValue = *max_it;
    bound = (maxValue-minValue)*range;
    std::ostringstream stream;
    
    size2 = input_data.size();
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::string valueStr = stream.str();
    stream << std::defaultfloat << bound;  // 使用默认的浮点表示
    std::string cpfilename = "compressed_data/compressed_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".sz";
    std::string decpfilename = "decompressed_data/decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string fix_path = "decompressed_data/fixed_decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string command;
    cout<<decpfilename<<endl;
    cout<<bound<<", "<<std::to_string(bound)<<endl;
    int result;
    if(compressor_id=="sz3"){
        
        command = "sz3 -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -M "+"ABS "+std::to_string(bound)+" -a";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    else if(compressor_id=="zfp"){
        cpfilename = "compressed_data/compressed_"+filename+"_"+std::to_string(bound)+".zfp";
        // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
        command = "zfp -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -a "+std::to_string(bound)+" -s";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    
    
    decp_data = getdata2(decpfilename);
    auto ends_t = std::chrono::high_resolution_clock::now();
    // cudaMemcpyFromSymbol(&h_s, decp_data, num1*sizeof(double), 0, cudaMemcpyDeviceToHost);
    std::chrono::duration<double> duration = ends_t - start;
    double compression_time = duration.count();
    std::vector<double> decp_data_copy(decp_data);
    
    or_direction_as.resize(size2);
    or_direction_ds.resize(size2);
    de_direction_as.resize(size2);
    de_direction_ds.resize(size2);
    or_label.resize(size2*2,-1);
    dec_label.resize(size2*2,-1);
    d_deltaBuffer.resize(size2,-2000.0);
    
    std::atomic<int>* id_array = new std::atomic<int>[size2];
    
    initializeWithIndex(or_label,or_direction_ds,or_direction_as);
    
   
    // dec_label.resize(size2*2);
    initializeWithIndex(dec_label,de_direction_ds,de_direction_as);
    
    adjacency.resize(size2*12, -1);
    std::vector<int> t1(size2,0);
    lowGradientIndices = find_low();
    // lowGradientIndices = t1;
    
    
    computeAdjacency();
    
    all_max.resize(size2);
    all_min.resize(size2);
    
    std::random_device rd;  // 随机数生成器
    std::mt19937 gen(rd()); // 以随机数种子初始化生成器
    
    
    
    
    auto searchtime = 0.0;
    auto fixtime_cp = 0.0;
    auto searchdirection_time = 0.0;
    auto fixtime = 0.0;
    auto fixtime_path = 0.0;
    auto getfcp = 0.0;
    auto getfpath = 0.0;
    auto mappath_path= 0.0;

    int cnt=0;
 
    std::vector<int> counter(4,0);  

    // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
    
    
    auto startt = std::chrono::high_resolution_clock::now();
    auto start1 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    
    
    // #pragma omp parallel for
    // for(int i=0;i<1;i++){
    start1 = std::chrono::high_resolution_clock::now();
    auto start_t = std::chrono::high_resolution_clock::now();
    find_direction(input_data, or_direction_as, or_direction_ds,1);
   
    // }
    
    
    
    
    
    // exit(0);
    // searchdirection_time = 0.0;
    
    
    // for(int i=0;i<10;i++){ 
    //     std::cout<<i<<std::endl;  
    find_direction(decp_data, de_direction_as, de_direction_ds);

    // }
    
    
    end = std::chrono::high_resolution_clock::now();
    
    duration = end - start1;
    // std::cout<<"1000cifindd: "<<duration.count()<<std::endl;
    
    searchdirection_time+=duration.count();
    counter[0]+=2;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    getfcp=0.0;
    // for(int i=0;i<10;i++){
    start1 = std::chrono::high_resolution_clock::now();
    get_false_criticle_points();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start1;
    getfcp+=duration.count();
    // }
    
    
    
    
    std::cout<<"getfcp: "<<duration.count()<<std::endl;
    
    // exit(0);
    getfcp+=duration.count();
    std::cout<<"10cigetfcp: "<<duration.count()<<std::endl;
    counter[1]+=1;
    mappath_path=0.0;
    
    // for(int i = 0; i < 0; i++)
    // {
       
    start1 = std::chrono::high_resolution_clock::now();
    mappath1(or_label, 1);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start1;
    mappath_path+=duration.count();
    // }
    
    
    
    
    std::cout<<"mappath: "<<mappath_path<<std::endl;
    
    // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
    // outFilef << "MSS_computation: "<<mappath_path <<std::endl;
        // outFilep << "fixtime_cp: "<<fixtime_cp << std::endl;
        // std::cout<<"1000direction: "<<searchdirection_time<<std::endl;
        // std::cout<<"1000getfcp: "<<getfcp<<std::endl;
        // exit(0);
        
    // return 0;
    // mappath_path+=duration.count();
    
    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start1;
    // mappath_path+=duration.count();
    // std::ofstream outFilep1("../result/performance_openmp_process_"+filename+"_"+std::to_string(bound)+"_"+compressor_id+".txt", std::ios::app);
    
    // // 检查文件是否成功打开
    // if (!outFilep1) {
    //     std::cerr << "Unable to open file for writing." << std::endl;
    //     return 1; // 返回错误码
    // }

    
    // outFilep1 << std::to_string(number_of_thread)<<":" << std::endl;
    // // outFilep << "duration: "<<duration.count() << std::endl;
    // outFilep1 << "getfcp: "<<getfcp << std::endl;
    // // outFilep << "fixtime_cp: "<<fixtime_cp << std::endl;
    // // outFilep << "fixtime_path: " << fixtime_path << std::endl;
    // outFilep1 << "mappath: " << mappath_path << std::endl;
    // outFilep1 << "finddirection: " << searchdirection_time << std::endl;
    // // outFilep << "getfpath:" << getfpath << std::endl;
    // // outFilep << "iteration number:" << ite+1 << std::endl;
    // // outFilep << "edit_ratio: "<<ratio << std::endl;
    // outFilep1 << "\n"<< std::endl;
    // return 0;
    start1 = std::chrono::high_resolution_clock::now();
    // for(int i=0;i<10;i++){
    // mappath1(dec_label, 0);
    // }
    
    end = std::chrono::high_resolution_clock::now();
    duration = end - start1;
    
    
    mappath_path+=duration.count();
    std::cout<<"mappath:"<<duration.count()<<std::endl;
    counter[2]+=2;

    start1 = std::chrono::high_resolution_clock::now();
    double right_labeled_ratio = 0.0;
    end = std::chrono::high_resolution_clock::now();
    duration = end - start1;
    getfpath+=duration.count();

    
    get_false_criticle_points();
    while (count_f_max>0 or count_f_min>0){
            std::cout<<count_f_max<<", "<<count_f_min<<std::endl;
            // cpite+=1;
            start1 = std::chrono::high_resolution_clock::now();
            initialization();
            #pragma omp parallel for

            for(auto i = 0; i < count_f_max; i ++){
                
                int critical_i = all_max[i];
                
                fix_maxi_critical(critical_i,0);

            }
                
                
                
            #pragma omp parallel for
            for(auto i = 0; i < count_f_min; i ++){

                int critical_i = all_min[i];
                fix_maxi_critical(critical_i,1);

            }
                
                applyDeltaBuffer();
                // counter[3]+=1;
                
                end = std::chrono::high_resolution_clock::now();
                duration = end-start1;
                fixtime_cp += duration.count();
                std::cout<<"fixfcp: "<<duration.count()<<std::endl;
                exit(0);
                
                
                start1 = std::chrono::high_resolution_clock::now();
                // find_direction(decp_data, de_direction_as, de_direction_ds);
                end = std::chrono::high_resolution_clock::now();
                duration = end-start1;
                searchdirection_time += duration.count();
                
                
                //finddirection:0, getfcp:1,  mappath2, fixcp:3
                start1 = std::chrono::high_resolution_clock::now();
                get_false_criticle_points();
                end = std::chrono::high_resolution_clock::now();
                duration = end-start1;
                getfcp += duration.count();
                find_direction(decp_data, de_direction_as, de_direction_ds);
                
        
    }
    mappath1(dec_label, 0);
    get_wrong_index_path();
    // while (count_f_max>0 or count_f_min>0){
    //     start1 = std::chrono::high_resolution_clock::now();
        
    //     #pragma omp parallel for

    //     for(auto i = 0; i < count_f_max; i ++){
            
    //         int critical_i = all_max[i];
            
    //         fix_maxi_critical(critical_i,0);

    //     }
        
        
        
    //     #pragma omp parallel for
    //     for(auto i = 0; i < count_f_min; i ++){

    //         int critical_i = all_min[i];
    //         fix_maxi_critical(critical_i,1);

    //     }
        
        
    //     counter[3]+=1;
        
    //     end = std::chrono::high_resolution_clock::now();
    //     duration = end-start1;
    //     fixtime_cp += duration.count();
        
    //     start1 = std::chrono::high_resolution_clock::now();
        
    //     find_direction(decp_data, de_direction_as, de_direction_ds);
    //     end = std::chrono::high_resolution_clock::now();
    //     duration = end-start1;
    //     searchdirection_time += duration.count();
    //     counter[0]+=1;
    //     //finddirection:0, getfcp:1,  mappath2, fixcp:3
    //     start1 = std::chrono::high_resolution_clock::now();
    //     get_false_criticle_points();
    
    //     duration = end-start1;
    //     getfcp += duration.count();
    //     counter[1]+=1;
    
    // }
    
     
    
    // start1 = std::chrono::high_resolution_clock::now();
       
       
    // // finddirection:0, getfcp:1,  mappath2, fixcp:3
    // mappath1(dec_label);
   
    
    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start1;
    // mappath_path += duration.count();
    // counter[2]+=1;
    // start1 = std::chrono::high_resolution_clock::now();
    // get_wrong_index_path();
    
    
    // end = std::chrono::high_resolution_clock::now();
    
    // duration = end - start1;
    // getfpath+=duration.count();
    
    std::vector<std::vector<double>> time_counter;
    
    while (wrong_index_as.size()>0 or wrong_index_ds.size()>0 or count_f_max>0 or count_f_min>0){
        searchtime = 0.0;
        fixtime_cp = 0.0;
        searchdirection_time = 0.0;
        fixtime = 0.0;
        fixtime_path = 0.0;
        getfcp = 0.0;
        getfpath = 0.0;
        mappath_path= 0.0;
        std::vector<double> temp_time;
        auto start_sub = std::chrono::high_resolution_clock::now();
        
        start1 = std::chrono::high_resolution_clock::now();
        std::cout<<count_f_max<<", "<<count_f_min<<","<<wrong_index_as.size()<<","<<wrong_index_ds.size()<<std::endl;
        int cpite = 0;
        fixtime_cp = 0.0;
        
        std::vector<double> decp_data_copy = decp_data;
        
        
        

        
        
        
       
        
        start1 = std::chrono::high_resolution_clock::now();
        initialization();

        #pragma omp parallel for
        for(int i =0;i< wrong_index_as.size();i++){
            int j = wrong_index_as[i];
            if(lowGradientIndices[j] == 1)
            {
                continue;
                }
            // std::cout<<j<<std::endl;
            fixpath(j,0,id_array);
        };
        
        // std::cout<<"danfjka1"<<std::endl;
        #pragma omp parallel for
        for(int i =0;i< wrong_index_ds.size();i++){
            int j = wrong_index_ds[i];
            if(lowGradientIndices[j] == 1){
                continue;}
            
            fixpath(j,1,id_array);
        };
        
        
        applyDeltaBuffer();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start1;
        fixtime_path += duration.count();
        
        // finddirection:0, getfcp:1,  mappath2, fixcp:3
        start1 = std::chrono::high_resolution_clock::now();
        find_direction(decp_data, de_direction_as, de_direction_ds);
        end = std::chrono::high_resolution_clock::now();
        duration = end-start1;
        searchdirection_time+=duration.count();
        
        start1 = std::chrono::high_resolution_clock::now();
        get_false_criticle_points();
        end = std::chrono::high_resolution_clock::now();
        duration = end-start1;
        getfcp+=duration.count();
        
        while (count_f_max>0 or count_f_min>0){
                // std::cout<<count_f_max<<", "<<count_f_min<<std::endl;
                cpite+=1;
                start1 = std::chrono::high_resolution_clock::now();
                initialization();
                #pragma omp parallel for

                for(auto i = 0; i < count_f_max; i ++){
                    
                    int critical_i = all_max[i];
                    
                    fix_maxi_critical(critical_i,0);

                }
                
                
                
                #pragma omp parallel for
                for(auto i = 0; i < count_f_min; i ++){

                    int critical_i = all_min[i];
                    fix_maxi_critical(critical_i,1);

                }
                
                applyDeltaBuffer();
                // counter[3]+=1;
                
                end = std::chrono::high_resolution_clock::now();
                duration = end-start1;
                fixtime_cp += duration.count();
                // std::cout<<"fixfcp: "<<duration.count()<<std::endl;
                // exit(0);
                
                
                start1 = std::chrono::high_resolution_clock::now();
                find_direction(decp_data, de_direction_as, de_direction_ds);
                end = std::chrono::high_resolution_clock::now();
                duration = end-start1;
                searchdirection_time += duration.count();
                
                
                //finddirection:0, getfcp:1,  mappath2, fixcp:3
                start1 = std::chrono::high_resolution_clock::now();
                get_false_criticle_points();
                end = std::chrono::high_resolution_clock::now();
                duration = end-start1;
                getfcp += duration.count();
                
        
            }
        
        start1 = std::chrono::high_resolution_clock::now();
        mappath1(dec_label);
        end = std::chrono::high_resolution_clock::now();
        duration = end-start1;
        mappath_path += duration.count();
        
        
        
        start1 = std::chrono::high_resolution_clock::now();
        get_wrong_index_path();
        end = std::chrono::high_resolution_clock::now();
        duration = end-start1;
        getfpath+=duration.count();
        
        
        temp_time.push_back(searchdirection_time);
        temp_time.push_back(getfcp);
        temp_time.push_back(fixtime_cp);
        temp_time.push_back(mappath_path);
        temp_time.push_back(getfpath);
        temp_time.push_back(fixtime_path);
        // temp_time_ratio.push_back(fixtime_path/whole);
        
        // temp_time_ratio.push_back(searchdirection_time/whole);
        
        // temp_time_ratio.push_back(searchtime/whole);
        
        
        // temp_time_ratio.push_back(get_path/whole);
        // record1.push_back(temp_time);
        // record_ratio.push_back(temp_time_ratio);
        // auto end_sub = std::chrono::high_resolution_clock::now();
        // duration = end_sub-start_sub;
        // temp_time.push_back(duration.count());
        temp_time.push_back(cpite);
        time_counter.push_back(temp_time);
        ite+=1;
    };
    end = std::chrono::high_resolution_clock::now();
    duration = end - start_t;
    double additional_time = duration.count();
    std::ofstream outFilepf("result/performance1_omp_"+std::to_string(bound)+"_"+".txt", std::ios::app);
    // 检查文件是否成功打开
    if (!outFilepf) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 0; // 返回错误码
    }
    
    int c1 = 0;  
    for (const auto& row : time_counter) {
        outFilepf << "iteration: "<<c1<<": ";
        for (size_t i = 0; i < row.size(); ++i) {
            outFilepf << row[i];
            if (i != row.size() - 1) { // 不在行的末尾时添加逗号
                outFilepf << ", ";
            }
        }
        // 每写完一行后换行
        outFilepf << std::endl;
        c1+=1;
    }
    outFilepf << "compression_time:" <<compression_time<<" additional_time:" <<additional_time << std::endl;
    outFilepf << "\n"<< std::endl;
    outFilepf.close();
    exit(0);
    // cout<<"出发"<<endl;
    end = std::chrono::high_resolution_clock::now();

    duration = end - startt;
    
    std::vector<double> indexs;
    std::vector<double> edits;
    for (int i=0;i<input_data.size();i++){
        
        if (decp_data_copy[i]!=decp_data[i]){
            indexs.push_back(i);
            edits.push_back(decp_data[i]-decp_data_copy[i]);
            
        }
    }
    
    double ratio = double(indexs.size())/(decp_data_copy.size());
    std::cout << "number_of_threads: " << number_of_thread << " seconds" << std::endl;
    std::cout << "duration: " << duration.count() << " seconds" << std::endl;
    std::cout << "get_fcp: " << getfcp << " seconds" << std::endl;
    std::cout << "fixtime_cp: " << fixtime_cp << " seconds" << std::endl;
    std::cout << "fixtime_path: " << fixtime_path << " seconds" << std::endl;
    std::cout << "searchtime: " << mappath_path << " seconds" << std::endl;
    std::cout << "finddirection: " << searchdirection_time << " seconds" << std::endl;
    std::cout << "getfpath:" << getfpath << std::endl;
    std::cout << "iteration number:" << ite << std::endl;
    exit(0);
    std::ofstream outFilep("result/performance2_openmp_1_"+filename+"_"+std::to_string(bound)+"_"+compressor_id+".txt", std::ios::app);
    // 检查文件是否成功打开
    if (!outFilep) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1; // 返回错误码
    }
    // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
    outFilep << std::to_string(number_of_thread)<<":" << std::endl;
    outFilep << "duration: "<<duration.count() << std::endl;
    outFilep << "getfcp: "<<getfcp << std::endl;
    outFilep << "ave_getfcp: "<<getfcp/counter[1] << std::endl;
    outFilep << "fixtime_cp: "<<fixtime_cp << std::endl;
    outFilep << "ave_fixtime_cp: "<<fixtime_cp/counter[3] << std::endl;
    outFilep << "fixtime_path: " << fixtime_path << std::endl;
    outFilep << "mappath: " << mappath_path << std::endl;
    outFilep << "ave_mappath: " << mappath_path/counter[2] << std::endl;
    outFilep << "finddirection: " << searchdirection_time << std::endl;
    outFilep << "ave_finddirection: " << searchdirection_time/counter[0] << std::endl;
    outFilep << "getfpath:" << getfpath << std::endl;
    outFilep << "iteration number:" << ite+1 << std::endl;
    outFilep << "edit_ratio: "<<ratio << std::endl;  
    int c = 0;  
    for (const auto& row : time_counter) {
        outFilep << "iteration: "<<c<<": ";
        for (size_t i = 0; i < row.size(); ++i) {
            outFilep << row[i];
            if (i != row.size() - 1) { // 不在行的末尾时添加逗号
                outFilep << ", ";
            }
        }
        // 每写完一行后换行
        outFilep << std::endl;
        c+=1;
    }
    outFilep << "\n"<< std::endl;
    
    
    
    return 0;
    std::ofstream outFile(fix_path);
    // std::ofstream outFile("values.bin", std::ios::binary | std::ios::out);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(decp_data.data()), decp_data.size() * sizeof(double));
    }
    outFile.close();
    // start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> diffs;  // 存储差值的 vector
    std::string indexfilename = "data"+filename+".bin";
    std::string editsfilename = "data_edits"+filename+".bin";
    std::string compressedindex = "data"+filename+".bin.zst";
    std::string compressededits = "data_edits"+filename+".bin.zst";
    // 在计算差值前将第一个元素添加到 diffs 中
    if (!indexs.empty()) {
        diffs.push_back(indexs[0]);
    }
    for (size_t i = 1; i < indexs.size(); ++i) {
        diffs.push_back(indexs[i] - indexs[i - 1]);
    }
    
    std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(diffs.data()), diffs.size() * sizeof(int));
    }
    file.close();
    command = "zstd -f " + indexfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    
    std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
    if (file1.is_open()) {
        file1.write(reinterpret_cast<const char*>(edits.data()), edits.size() * sizeof(double));
    }
    file1.close();
    command = "zstd -f " + editsfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    // compute result
    double compressed_indexSize = fs::file_size(compressedindex);
    double compressed_editSize = fs::file_size(compressededits);
    double original_indexSize = fs::file_size(indexfilename);
    double original_editSize = fs::file_size(editsfilename);
    double original_dataSize = fs::file_size(inputfilename);
    double compressed_dataSize = fs::file_size(cpfilename);
    // std::cout<<compressed_indexSize<<", "<<original_indexSize<<", "<<original_indexSize/compressed_indexSize<<std::endl;
    // std::cout<<compressed_editSize<<", "<<original_editSize<<", "<<original_editSize/compressed_editSize<<std::endl;
    // std::cout<<compressed_dataSize<<", "<<original_dataSize<<", "<<original_dataSize/compressed_dataSize<<std::endl;
    double overall_ratio = (original_indexSize+original_editSize+original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    double bitRate = 64/overall_ratio; 
    
    


    double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);
    std::cout<<psnr<<", "<<fixed_psnr<<std::endl;
    std::cout<<"right: "<<right_labeled_ratio<<std::endl;
    std::ofstream outFile3("../result/result_"+filename+"_"+compressor_id+".txt", std::ios::app);

    // 检查文件是否成功打开
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1; // 返回错误码
    }

    
    outFile3 << std::to_string(bound)<<":" << std::endl;
    
    outFile3 << "OCR: "<<overall_ratio << std::endl;
    outFile3 << "CR: "<<original_dataSize/compressed_dataSize << std::endl;
    outFile3 << "OBR: "<<bitRate << std::endl;
    outFile3 << "BR: "<< (compressed_dataSize*8)/size2 << std::endl;
    outFile3 << "psnr: "<<psnr << std::endl;
    outFile3 << "fixed_psnr: "<<fixed_psnr << std::endl;

    outFile3 << "right_labeled_ratio: "<<right_labeled_ratio << std::endl;
    outFile3 << "edit_ratio: "<<ratio << std::endl;
    outFile3 << "\n" << std::endl;
    // 关闭文件
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;

    
    
    if(mode==1){
        if(bitRate==target_br){
            std::string path1 ="decompressed_data/fixed_decp_"+filename+"_"+std::to_string(bound)+"target_"+std::to_string(bitRate)+".bin";
            std::ofstream outFile4(path1);
            // std::ofstream outFile("values.bin", std::ios::binary | std::ios::out);
            if (outFile4.is_open()) {
                outFile4.write(reinterpret_cast<const char*>(decp_data.data()), decp_data.size() * sizeof(double));
            }
            outFile4.close();
            std::ofstream outFile5("../result/result_"+filename+"_"+compressor_id+".txt", std::ios::app);

            // 检查文件是否成功打开
            if (!outFile5) {
                std::cerr << "Unable to open file for writing." << std::endl;
                return 1; // 返回错误码
            }

            
            outFile5 << filename <<"_"<<std::to_string(bound)<<":" << std::endl;
            
            outFile5 << "OCR: "<<overall_ratio << std::endl;
            outFile5 << "CR: "<<original_dataSize/compressed_dataSize << std::endl;
            outFile5 << "OBR: "<<bitRate << std::endl;
            outFile5 << "BR: "<< (compressed_dataSize*8)/size2 << std::endl;
            outFile5 << "psnr: "<<psnr << std::endl;
            outFile5 << "fixed_psnr: "<<fixed_psnr << std::endl;

            outFile5 << "right_labeled_ratio: "<<right_labeled_ratio << std::endl;
            outFile5 << "edit_ratio: "<<ratio << std::endl;
            outFile5 << "\n" << std::endl;
            // 关闭文件
            outFile5.close();
            std::cout << "target_bitrate_founded" << std::endl;
        }
    }
    return 0;
}
