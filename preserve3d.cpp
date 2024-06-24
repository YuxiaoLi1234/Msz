// #include "cuda_runtime.h"
// #include "cublas_v2.h"
// #include <cuda_runtime.h>
// #include "device_launch_parameters.h"
// #include <fstream>
// #include <sstream>
// #include <vector>
// #include <cstdlib>
// #include <fstream>
// #include <stdatomic.h>
// #include <parallel/algorithm>  
// #include <unordered_map>
// #include <random>
// #include <atomic>
// #include <string>
// #include <omp.h>
// #include <iostream>
// #include <unordered_set>
// #include <set>
// #include <map>
// #include <algorithm>
// #include <numeric>
// #include <utility>
// #include <fstream>
// #include <iomanip>
// #include <chrono>
// #include <random>
// #include <iostream>
// #include <filesystem>
// // #include <vtkSmartPointer.h>
// // #include <vtkImageData.h>
// // #include <vtkIntArray.h>
// // #include <vtkPointData.h>
// // #include <vtkXMLImageDataReader.h>
// // #include <vtkXMLImageDataWriter.h>
// using namespace std;
// namespace fs = std::filesystem;
// int pre = 0;
// // g++ -std=c++17 -O3 -g hello.cpp -o helloworld
// // g++-12 -std=c++17 -O3 -g -fopenmp -c preserve3d.cpp -o hello2.o
// // g++-12 -fopenmp hello2.o kernel.o -lcudart -o helloworld
// // g++-12 -fopenmp -std=c++17 -O3 -g hello2.cpp -o helloworld2
// // 4.5177893520000003
// // g++ -fopenmp hello2.o kernel.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld

// int width;
// int height;
// int depth;
// int size2;
// int un_sign_as;
// int un_sign_ds;
// int ite = 0;
// std::vector<int> all_max, all_min, all_d_max, all_d_min;
// atomic_int count_max = 0;
// atomic_int count_min = 0;
// atomic_int count_f_max = 0;
// atomic_int count_f_min = 0;

// std::vector<int> record;
// std::vector<std::vector<float>> record1;
// std::vector<std::vector<float>> record_ratio;
// // 没有0，0 和 1，1
// std::vector<std::vector<int>> directions1 = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};

// // std::vector<double> getdata(std::string filename){
    
// //      std::vector<double> data;
// //      data.resize(width*height*depth);
// //      std::string line;
// //      std::ifstream file(filename);
// //     //  std::ifstream file(filename);
// //     // std::cout<<"dh"<<std::endl;
// //     // std::vector<double> data;
// //     double num;

// //     if (!file.is_open()) {
// //         std::cerr << "Error opening file" << std::endl;
// //         return data;
// //     }
// //     int cnt=0;
// //     while (!file.eof()) { // 循环直到文件结束
// //         if (file >> num) {
// //             data[cnt] = num;
// //             cnt+=1;
// //         } 
// //         else {
            
// //             file.clear(); // 清除失败的状态
// //             std::string invalid_input;
// //             if (file >> invalid_input) {
// //                 std::cout << "无效的输入: " << invalid_input << std::endl;
// //             }
// //             // file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略当前行的剩余部分
// //             if (file.eof()) break;
// //         }
// //     }
// //     // std::cout<<"读完了"<<cnt<<std::endl;
// //     file.close();
    
// //     // std::cout<<data.size()<<std::endl;
// //     // Optionally print the data to verify
// //     // for (double num : data) {
// //     //     std::cout << num << " ";
// //     // }
// //     // std::cout << std::endl;

// //      // 读取每一行
// //     //  while (std::getline(file, line)) {
// //     //       std::vector<double> row;
// //     //       std::stringstream ss(line);
// //     //       std::string value;

// //     //       // 分割行中的每个值
// //     //       while (std::getline(ss, value, ',')) {
// //     //         try {
// //     //             double number = std::stod(value);
// //     //             // 后续操作
// //     //         } catch (const std::exception& e) {
// //     //             if(typeid(value) == typeid(std::string)){
// //     //                 std::cout<<value<<std::endl;
// //     //             }
                
// //     //             std::cerr << "转换错误: " << e.what() << '\n';
// //     //             // 异常处理代码
// //     //         }

// //     //         // std::cout<<value<<std::endl;
// //     //            data.push_back(std::stod(value)); // 将字符串转换为 double
// //     //       }

// //     //     //   data.push_back(row);
// //     //  }

// //     //  file.close();
     
// //      return data;
// // };
// // std::vector<double> getdata1(std::string filename){
// //     std::vector<double> myVector;

// //     // 打开一个文件流
// //     std::ifstream inFile("output.txt");

// //     // 检查文件是否成功打开
// //     if (inFile.is_open()) {
// //         double value;
// //         // 读取文件中的每个整数，并添加到 vector 中
// //         while (!inFile.eof()) { // 循环直到文件结束
// //             if (inFile >> value) {
// //                 myVector.push_back(value);
                
// //             } 
// //             else {
                
// //                 inFile.clear(); // 清除失败的状态
// //                 std::string invalid_input;
// //                 if (inFile >> invalid_input) {
// //                     std::cout << "无效的输入: " << invalid_input << std::endl;
// //                 }
// //                 // file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略当前行的剩余部分
// //                 if (inFile.eof()) break;
// //             }
// //         }
        
        
// //         // 关闭文件流
// //         inFile.close();
// //     } 
    
    
    
// //     return myVector;
// // }
// // std::string inputfilename= "result_3d.txt";

// std::vector<double> getdata2(std::string filename){
//     std::ifstream file(filename, std::ios::binary | std::ios::ate);
//     std::vector<double> data;
//     if (!file) {
//         std::cerr << "无法打开文件" << std::endl;
//         return data;
//     }

//     // 获取文件大小
//     std::streamsize size = file.tellg();
//     file.seekg(0, std::ios::beg);

//     // 确定文件中有多少个float值
//     std::streamsize num_floats = size / sizeof(double);
//     // std::cout<<num_floats<<std::endl;
//     // 创建一个足够大的vector来存储文件内容
//     std::vector<double> buffer(num_floats);

//     // 读取文件内容到vector
//     if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
//         // std::cout << "文件读取成功" << std::endl;

//         // 输出读取的数据，以验证
//         // for (double num : buffer) {
//         //     std::cout << num << " ";
//         // }
//         // std::cout << std::endl;
//         return buffer;
//     } else {
//         std::cerr << "文件读取失败" << std::endl;
//         return buffer;
//     }
    
//     return buffer;
// }
// std::string inputfilename;
// // std::string decompfilename= "result_3d_decomp.txt";
// // std::string decompfilename= "output.txt";
// std::string decompfilename;

// // cnpy::NpyArray arr = cnpy::npy_load("result3d.npy");
// // float* loaded_data = arr.data<float>();






// // std::vector<int> ordered_input_data = processdata(input_data);

// std::vector<double> decp_data;
// std::vector<double> input_data;
// // std::vector<double> bound_data;
// std::unordered_map<int, double> maxrecord;
// std::unordered_map<int, double> minrecord;
// double bound;
// // std::vector<int> findUniqueElements(const std::unordered_set<int>& set1, const std::unordered_set<int>& set2) {
// //     std::vector<int> uniqueElements;
// //     // 检查 set1 中有而 set2 中没有的元素
// //     for (const auto& elem : set1) {
// //         if (set2.find(elem) == set2.end()) {
// //             uniqueElements.push_back(elem);
// //         }
// //     }

// //     // 检查 set2 中有而 set1 中没有的元素
// //     for (const auto& elem : set2) {
// //         if (set1.find(elem) == set1.end()) 
// //         {
// //             uniqueElements.push_back(elem);
// //         }
// //     }

// //     return uniqueElements;
// // }


// std::vector<int> find_low(){
//     std::vector<int> lowGradientIndices(size2, 0);
    
//     const double threshold = 1e-16; // 梯度阈值
//     // 遍历三维数据计算梯度
//     for (int i = 0; i < width; ++i) {
        
//         for (int j = 0; j < height; ++j) {
            
//             for (int k = 0; k < depth; ++k) {
                
//                 int rm = i  + j * width + k * (height * width);
                
                
//                 // for(int q=0;q<12;q++){
//                 for (auto& dir : directions1) {
//                     int newX = i + dir[0];
//                     int newY = j + dir[1];
//                     int newZ = k + dir[2];
//                     int r = newX  + newY * width + newZ* (height * width);
//                     if(r>=0 and r<size2){
//                         double gradZ3 = abs(input_data[r] - input_data[rm])/2;
//                         if (gradZ3<=threshold) {
//                             lowGradientIndices[rm]=1;
//                             lowGradientIndices[r]=1;
//                         }
//                     // }
//                 }
//                 }
//             }
//         }
//     }
//     // std::cout<<"ok"<<std::endl;
//     return lowGradientIndices;
// }

// std::vector<int> lowGradientIndices;

// std::vector<std::vector<int>> _compute_adjacency(){
//     std::vector<std::vector<int>> adjacency;
//     for (int i = 0; i < size2; ++i) {
//             int y = (i / (width)) % height; // Get the x coordinate
//             int x = i % width; // Get the y coordinate
//             int z = (i / (width * height)) % depth;
//             std::vector<int> adjacency_temp;
//             for (auto& dir : directions1) {
//                 int newX = x + dir[0];
//                 int newY = y + dir[1];
//                 int newZ = z + dir[2];
//                 int r = newX + newY  * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                
                
//                 // Check if the new coordinates are within the bounds of the mesh
//                 if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0 && lowGradientIndices[r] != 1) {
                    
//                     adjacency_temp.push_back(r);
//                 }
//                 // if(input_data[r]-input_data[i]==0 and input_data[r]==0){
//                 //     continue;
//                 // }
//             }
//             adjacency.push_back(adjacency_temp);
//         }
//     return adjacency;
// }
// std::vector<std::vector<int>> adjacency;

// // std::vector<double> add_noise(const std::vector<double>& data, double x) {
// //     std::vector<double> noisy_data = data;
// //     std::random_device rd;
// //     std::mt19937 gen(rd());
// //     std::uniform_real_distribution<> dis(-x, x);

// //     std::for_each(noisy_data.begin(), noisy_data.end(), [&dis, &gen](double &d){
// //         d += dis(gen);
// //     });

// //     return noisy_data;
// // };

// std::map<std::tuple<int, int, int>, int> createDirectionMapping() {
//     std::map<std::tuple<int, int, int>, int> direction_mapping_3d;
//     direction_mapping_3d[std::make_tuple(0,1,0)] = 1;
//     direction_mapping_3d[std::make_tuple(0,-1,0)] = 2;
//     direction_mapping_3d[std::make_tuple(1,0,0)] = 3;
//     direction_mapping_3d[std::make_tuple(-1,0,0)] = 4;
//     direction_mapping_3d[std::make_tuple(-1,1,0)] = 5;
//     direction_mapping_3d[std::make_tuple(1,-1,0)] = 6;

//     // Additional 3D directions
//     direction_mapping_3d[std::make_tuple(0, 0, -1)] = 7;   // down in Z
//     direction_mapping_3d[std::make_tuple(0,-1, 1)] = 8;   // down-left in Z
//     direction_mapping_3d[std::make_tuple(0, 0, 1)] = 9;    // up in Z
//     direction_mapping_3d[std::make_tuple(0, 1, -1)] = 10;  // up-right in Z
//     direction_mapping_3d[std::make_tuple(-1, 0, 1)] = 11;  // left-up in Z
//     direction_mapping_3d[std::make_tuple(1, 0, -1)] = 12; 

//     return direction_mapping_3d;
// };

// int a = 0;
// // std::vector<int> processdata(std::vector<double> data){
// //     std::vector<std::pair<double, int>> value_index_pairs;
    
// //     int n = data.size();
    
// //     // value_index_pairs.clear();
    
// //     for (int i = 0; i < n; ++i) {
// //         value_index_pairs.emplace_back(data[i], i);
// //     }

// //     __gnu_parallel::stable_sort(value_index_pairs.begin(), value_index_pairs.end(),[](const std::pair<double, int>& a, const std::pair<double, int>& b) {
// //             if (a.first == b.first) {
// //                 return a.second < b.second; // 注意这里是‘>’
// //             }
// //             return a.first < b.first;
// //     });
    
    
// //     std::vector<int> sorted_indices(n);
// //     // #pragma omp parallel for
// //     for (int i = 0; i < n; ++i) {
        
// //         sorted_indices[value_index_pairs[i].second] = i;
// //     }
    
// //     return sorted_indices;

// // };
// // std::string inputfilename= "input_arrayc"+std::to_string(width)+".csv";


// // std::vector<double> decp_data = add_noise(input_data,bound);

// std::map<int, std::tuple<int, int, int>> createReverseMapping(const std::map<std::tuple<int, int ,int>, int>& originalMap) {
//     std::map<int, std::tuple<int, int, int>> reverseMap;
//     for (const auto& pair : originalMap) {
//         reverseMap[pair.second] = pair.first;
//     }
//     return reverseMap;
// }
// std::map<std::tuple<int, int ,int>, int> direction_mapping = createDirectionMapping();

// std::map<int, std::tuple<int, int ,int>> reverse_direction_mapping = createReverseMapping(direction_mapping);
// class myTriangularMesh {
//     public:
//         std::vector<double> values;
//         // std::vector<std::vector<int>> adjacency;
//         std::unordered_set<int> maxi;
//         std::unordered_set<int> mini;
//         // 函数声明

//         // void _compute_adjacency();
//         void get_criticle_points();
//         // std::vector<int> adjacent_vertices( int index);

// }; 

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

// // void myTriangularMesh::get_criticle_points(){
// //     std::vector<int> maxi_temp;
// //     std::vector<int> mini_temp;
// //     std::vector<int> maximum(size2, 0);
// //     std::vector<int> minimum(size2, 0);
    
   
// //     for (int i = 0; i < size2; ++i) {
// //             bool is_maxima = true;
// //             bool is_minima = true;

            
// //             for (int j : adjacency[i]) {
// //                 if (this->values[j] > this->values[i]) {
// //                     is_maxima = false;
// //                     break;
// //                 }
// //                 if(this->values[j] == this->values[i] and j>i){
// //                     is_maxima = false;
// //                     break;
// //                 }
// //             }
// //             for (int j : adjacency[i]) {
// //                 if (this->values[j] < this->values[i]) {
// //                     is_minima = false;
// //                     break;
// //                 }
// //                 if(this->values[j] == this->values[i] and j<i){
// //                     is_minima = false;
// //                     break;
// //                 }
// //             }
    

            

// //             if (is_maxima) {
// //                 maximum[i]=-1; // Add index to maxima list
// //             }
// //             if (is_minima) {
// //                 minimum[i]=-1; // Add index to minima list
// //             }
// //         }
    
    
    

// //     // #pragma omp parallel for
// //         for (int i = 0; i < size2; ++i) {
// //             if (maximum[i] == -1) {
// //                 // #pragma omp critical
// //                 maxi_temp.push_back(i);
// //             }
// //         }
// //     // #pragma omp parallel for
// //         for (int i = 0; i < size2; ++i) {
// //             if (minimum[i] == -1) {
// //                 // #pragma omp critical
// //                 mini_temp.push_back(i);
// //             }
// //         }
// //     std::unordered_set<int> mesh_maxi(maxi_temp.begin(), maxi_temp.end());
// //     std::unordered_set<int> mesh_mini(mini_temp.begin(), mini_temp.end());
// //     this->maxi = mesh_maxi;
// //     this->mini = mesh_mini;
// // };



// // std::vector<double> linspace(double start, double end, int num) {
// //     std::vector<double> linspaced;

// //     if (num == 0) { 
// //         return linspaced; 
// //     }
// //     if (num == 1) {
// //         linspaced.push_back(start);
// //         return linspaced;
// //     }

// //     double delta = (end - start) / (num - 1);

// //     for(int i=0; i < num-1; ++i) {
// //         linspaced.push_back(start + delta * i);
// //     }
// //     linspaced.push_back(end); 
    
// //     // 确保end被包括进去
// //     return linspaced;
// // }


// // std::vector<int> ordered_decp_data = processdata(decp_data);

// myTriangularMesh or_mesh;
// myTriangularMesh de_mesh;
// std::vector<int> wrong_maxi_cp;
// std::vector<int> wrong_min_cp;
// std::vector<int> wrong_index_as;
// std::vector<int> wrong_index_ds;
// int direction_to_index_mapping[12][3] = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};

// int getDirection(const std::map<std::tuple<int, int, int>, int>& direction_mapping, int row_diff, int col_diff, int dep_diff) {
//     auto it = direction_mapping.find(std::make_tuple(row_diff, col_diff,dep_diff));
    
//     if (it != direction_mapping.end()) {
//         return it->second;
//     } else {
//         return -1; 
//     }
// }
// // int from_direction_to_index(int cur, int direc){
// //     if (direc==-1) return cur;
// //     int x = cur % width;
// //     int y = (cur / width) % height;
// //     int z = (cur/(width * height))%depth;
    
// //     auto it = reverse_direction_mapping.find(direc);
// //     if (it != reverse_direction_mapping.end()) {
// //     // 找到了元素，it->first 是键，it->second 是值
// //         std::tuple<int, int,int> value = it->second;
        
// //         int delta_row = std::get<0>(value); 
// //         int delta_col = std::get<1>(value); 
// //         int delta_dep = std::get<2>(value); 
// //         int next_row = x + delta_row;
// //         int next_col = y + delta_col;
// //         int next_dep = z + delta_dep;
        
// //         return next_row + next_col  * width + next_dep* (height * width);
// //     } else {
// //         return -1;
// //     }
    
// // };
// int from_direction_to_index(int cur, int direc){
    
//     if (direc==-1) return cur;
//     int x = cur % width;
//     int y = (cur / width) % height;
//     int z = (cur/(width * height))%depth;
//     // printf("%d %d\n", row, rank1);
//     if (direc >= 1 && direc <= 12) {
//         int delta_row = direction_to_index_mapping[direc-1][0];
//         int delta_col = direction_to_index_mapping[direc-1][1];
//         int delta_dep = direction_to_index_mapping[direc-1][2];
        
        
//         int next_row = x + delta_row;
//         int next_col = y + delta_col;
//         int next_dep = z + delta_dep;
//         // printf("%d \n", next_row * width + next_col);
//         // return next_row * width + next_col + next_dep* (height * width);
//         return next_row + next_col * width + next_dep* (height * width);
//     }
//     else {
//         return -1;
//     }
//     // return 0;
// };
// // int find_direction (int index,  std::vector<double> &data ,int direction){
    
   
// //     double mini = 0;
// //     int size1 = adjacency[index].size();
    
// //     std::vector<int> indexs = adjacency[index];
// //     int largetst_index = index;
    
// //     if (direction == 0){
        
// //         for(int i =0;i<size1;++i){
// //             if(lowGradientIndices[indexs[i]] == 1){
// //                 continue;
// //             }
// //             if((data[indexs[i]]>data[largetst_index] or (data[indexs[i]]==data[largetst_index] and indexs[i]>largetst_index)) and lowGradientIndices[indexs[i]] == 0){
// //                 mini = data[indexs[i]]-data[index];
                
// //                 largetst_index = indexs[i];
// //                 // }
                
// //             };
// //         };
        
        
// //     }
// //     else{
// //         for(int i =0;i<size1;++i){
// //             if(lowGradientIndices[indexs[i]] == 1){
// //                 continue;
// //             }
// //             if((data[indexs[i]]<data[largetst_index] or (data[indexs[i]]==data[largetst_index] and indexs[i]<largetst_index)) and lowGradientIndices[indexs[i]] == 0){
                
// //                 mini = data[indexs[i]]-data[index];
                
// //                 largetst_index = indexs[i];
// //                 // 113618, 113667, 111168, 111168
// //                 // if(wrong_index_ds.size()==1 and index==111168){
// //                 //     std::cout<<index<<", "<<largetst_index<<","<<decp_data[largetst_index]<<", "<<decp_data[113667]<<std::endl;
// //                 // }

                
// //             };
// //         };
// //     };
    
// //     // if(lowGradientIndices[largest_index] == 1){
// //     //     std::cout<<largetst_index<<std::endl;
// //     // }
// //     // int row_l = (largetst_index % (height * width)) / width;
// //     // int row_i = (index % (height * width)) / width;
    
// //     // int row_diff = row_l - row_i;
// //     // int col_diff = (largetst_index % width) - (index % width);

// //     // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
// //     int row_l = (largetst_index / (height)) % width;
// //     int row_i = (index / (height)) % width;
    
// //     int col_diff = row_l - row_i;
// //     int row_diff = (largetst_index % height) - (index % height);

// //     int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
// //     int d = getDirection(direction_mapping, row_diff, col_diff,dep_diff);
    
    
// //     return d;

// // };
// // int find_direction1 (int index,std::map<int,int> &data ,int direction){

// //     double mini = 0;
// //     int size1 = adjacency[index].size();
// //     std::vector<int> indexs = adjacency[index];
// //     int largetst_index;
    
// //     if (direction == 0){
        
// //         for(int i =0;i<size1;++i){
            
// //             if(data[indexs[i]]-data[index]>mini){
// //                 mini = data[indexs[i]]-data[index];
// //                 largetst_index = indexs[i];
// //             };
// //         };
        
        
// //     }
// //     else{
// //         for(int i =0;i<size1;++i){
// //             if(data[adjacency[index][i]]-data[index]<mini){
// //                 mini = data[adjacency[index][i]]-data[index];
// //                 largetst_index = adjacency[index][i];
// //             };
// //         };
// //     };
// //     // std::cout<<largetst_index<<","<<index<<std::endl;
// //     // int row_l = (largetst_index % (height * width)) / width;
// //     // int row_i = (index % (height * width)) / width;
    
// //     // int row_diff = row_l - row_i;
// //     // int col_diff = (largetst_index % width) - (index % width);

// //     // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
// //     int row_l = (largetst_index / (height)) % width;
// //     int row_i = (index / (height)) % width;
    
// //     int col_diff = row_l - row_i;
// //     int row_diff = (largetst_index % height) - (index % height);

// //     int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
// //     int d = getDirection(direction_mapping, row_diff, col_diff,dep_diff);
    
    
// //     return d;

// // };
// // std::vector<int> mappath(myTriangularMesh& mesh, std::vector<int> direction_as, std::vector<int> direction_ds){
// //     // std::vector<std::pair<int, int>> label(size2, std::make_pair(-1, -1));
// //     std::vector<int> label(size2*2, -1);

// //     #pragma omp parallel for
// //     for (int i = 0;i<size2;++i){
// //         if(lowGradientIndices[i] == 1){
// //             continue;
// //         }
        
// //         int cur = i;
        
// //         while (direction_as[cur]!=-1){
// //             // if(lowGradientIndices.find(cur)!=lowGradientIndices.end()){
// //             //     std::cout<<cur<<", "<<i<<std::endl;
// //             // }
// //             int direc = direction_as[cur];
            
// //             int rank1 = (cur / (height)) % width;
// //             int row1 = cur%height;
// //             int depth1 = cur/(width * height);
            
// //             int next_vertex = from_direction_to_index(cur, direc);
            
            
// //             cur = next_vertex;
            
// //             if (label[cur*2+1] != -1){
// //                 cur = label[cur*2+1];
// //                 break;
// //             }
            
// //         }

// //         if(direction_as[i]!=-1){
// //             label[i*2+1] = cur;
// //         }
// //         else{
// //             label[i*2+1] = -1;
// //         };
        
// //         cur = i;

// //         while (direction_ds[cur]!=-1){
            
// //             int direc = direction_ds[cur];
// //             int rank1 = (cur / (height)) % width;
// //             int row1 = cur%height;
// //             int depth1 = cur/(width * height);
            
// //             int next_vertex = from_direction_to_index(cur, direc);
            

// //             cur = next_vertex;

// //             if (label[cur*2+0] != -1){
// //                 cur = label[cur*2+0];
// //                 break;
// //             }
// //         }

// //         if(direction_ds[i]!=-1){
// //             label[i*2+0] = cur;
// //         }
// //         else{
// //             label[i*2+0] = -1;
// //         };
        
    
// //     };
    
// //     return label;
// // };
// std::vector<int> or_direction_as;
// std::vector<int> or_direction_ds;
// std::vector<int> de_direction_as;
// std::vector<int> de_direction_ds;

// // std::vector<std::pair<int, int>> or_label(size2, std::make_pair(-1, -1));
// std::vector<int> dec_label;
// std::vector<int> or_label;
// // std::vector<std::pair<int, int>> dec_label(size2, std::make_pair(-1, -1));
// // bool isNumberInArray(const std::vector<int>& array, int number) {
// //     for (int num : array) {
// //         if (num == number) {
// //             return true;  // 找到了，返回 true
// //         }
// //     }
// //     return false;  // 遍历完毕，未找到，返回 false
// // }

// // int fix_maxi_critical(int index, int direction){
    
// //     if (direction == 0){
        
// //         if (or_direction_as[index]!=-1){
            
// //             int next_vertex = from_direction_to_index(index,or_direction_as[index]);
// //             int smallest_vertex = next_vertex;
// //             double threshold = std::numeric_limits<double>::lowest();
            
            
// //             for(int i:adjacency[index]){
// //                 if(lowGradientIndices[i] == 1){
// //                     continue;
// //                 }
// //                 if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
// //                     smallest_vertex = i;
// //                     threshold = input_data[i];
// //                 }
// //             }
            
// //             threshold = decp_data[smallest_vertex];
// //             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
// //             double d = (decp_data[index] - input_data[index] + bound )/2.0;
// //             double d1 = ((input_data[next_vertex] + bound) - decp_data[next_vertex])/2.0;
// //             double diff1 = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            
// //             if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
// //                 de_direction_as[index]=or_direction_as[index];
            
// //                 return 0;
// //             }
            
// //             if(d>=1e-16 or d1>=1e-16){
                
// //                 if(decp_data[index]==decp_data[next_vertex])
// //                     {
                        
// //                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
// //                             d/=2;
// //                         }
// //                         if (abs(input_data[index]-decp_data[index]+d)<=bound){
// //                             decp_data[index] -= d/64;
// //                         }
// //                     }
// //                 else{
// //                     if(decp_data[index]>=decp_data[next_vertex]){
                        
// //                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
// //                                 d/=2;
// //                         }
                        
// //                         if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
// //                             while(decp_data[index] - d < threshold and d>=2e-16)
// //                             {
// //                                 d/=2;
// //                             }
                            
                            
// //                         }
// //                         // else if(threshold>decp_data[next_vertex]){
                            
                            
// //                         //     double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/64;
                            
// //                         //     if(diff2>=1e-16){
// //                         //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
// //                         //             diff2/=2;
// //                         //         }
                                
// //                         //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
// //                         //             if(smallest_vertex==66783){std::cout<<"在这里11."<<std::endl;}
// //                         //             decp_data[smallest_vertex]-=diff2;
// //                         //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
// //                         //         }
                                
                                
// //                         //     }
                            
// //                         // }

// //                         if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-16){
// //                             // if(index==1620477){
// //                             //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
// //                             //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
// //                             //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
// //                             // }
                            
// //                             decp_data[index] -= d;
                            
                            
                                            
// //                         }
// //                         // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
// //                         //     // if(index==1620477){
// //                         //     //     // std::cout<<"next_vertex: "<<decp_data[next_vertex]<<std::endl;
// //                         //     //     // std::cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<std::endl;
// //                         //     //     std::cout<<"before index: "<<decp_data[index]<<std::endl;
                                
// //                         //     // }
                            
// //                         //     decp_data[next_vertex] += d1;
                            
                            
                                            
// //                         // }
                        
                        
                   
// //                 };
// //                      }
            
                 
            
                
// //             }
// //             else{
                
// //                 if(decp_data[index]>decp_data[next_vertex]){
// //                     if(abs(input_data[index]-decp_data[next_vertex])<bound){
// //                             double t = (decp_data[next_vertex]-(input_data[index]-bound))/2.0;
// //                             decp_data[index] = decp_data[next_vertex] + t;
// //                             // decp_data[next_vertex] = t;
// //                         }
// //                     else{
// //                         decp_data[index] = input_data[index] - bound;
// //                         // decp_data[next_vertex] = input_data[next_vertex] + bound;
// //                     }
                    
// //                 }
// //                 else if(decp_data[index]==decp_data[next_vertex]){
// //                     double d = bound - (input_data[index]-decp_data[index])/64;
// //                     // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
// //                     //         d/=2;
// //                     // }
// //                     // if(index==157569){
// //                     //     std::cout<<"在这时候d: "<<d<<std::endl;
// //                     // }   
// //                     // double d = 1e-16;
// //                     if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
// //                         decp_data[index]-=d;
// //                     }
// //                     else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d)<=bound){
// //                         // if(next_vertex==78){std::cout<<"在这里21"<<std::endl;}
// //                         decp_data[next_vertex]+=d;
// //                     }
// //                 }
                
// //             }
            
            
        
// //         }
// //         else{
// //             // if(index==25026 and count_f_max<=770){
// //             //     std::cout<<"在这里"<<std::endl;
// //             // }
            
// //             int largest_index = from_direction_to_index(index,de_direction_as[index]);
// //             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
// //             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
// //             // if(index==25026 and count_f_max<=770){
// //             //     std::cout<<"改变前"<<std::endl;
// //             //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
// //             //     std::cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<std::endl;
// //             //     // std::cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<std::endl;
// //             //     std::cout<<"diff: "<<d<<","<<"d: "<<d<<std::endl;
// //             //     std::cout<<or_direction_as[25026]<<de_direction_as[25026]<<std::endl;
// //             // }
// //             if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
// //                 de_direction_as[index] = -1;
// //             }
// //             if(d>=1e-16){
                
// //                 if (decp_data[index]<=decp_data[largest_index]){
// //                     if(abs(input_data[largest_index]-decp_data[index]+d)){
// //                         // if(largest_index==66783){std::cout<<"在这里17"<<std::endl;}
// //                         decp_data[largest_index] = decp_data[index]-d;
// //                     }
// //                 }
                
            
                
// //             }
            
// //             else{
// //                 if(decp_data[index]<=decp_data[largest_index]){
                    
// //                     decp_data[index] = input_data[index] + bound;
// //                 }
                    
// //             }

            
            
// //         }
        
        
    
// //     }
    
// //     else{
        
// //         if (or_direction_ds[index]!=-1){
// //             int next_vertex= from_direction_to_index(index,or_direction_ds[index]);
            
// //             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
// //             double d =  (bound+input_data[index]-decp_data[index])/2.0;
// //             // double d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
// //             double d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
// //             if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
// //                 de_direction_ds[index]=or_direction_ds[index];
// //                 return 0;
// //             }

// //             // if(index == 6595 and count_f_min==5){
// //             //     std::cout<<"下降："<<std::endl;
// //             //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
// //             //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
// //             //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
// //             //     std::cout<<"diff: "<<diff<<std::endl;
// //             //     std::cout<<"d: "<<d<<std::endl;
// //             //     std::cout<<"d1: "<<d1<<std::endl;
// //             // }
            
// //             if(diff>=1e-16 or d>=1e-16 or d1>=1e-16){
                
// //                 if(decp_data[index]==decp_data[next_vertex]){
                    
                      
                    
// //                         while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
// //                             diff/=2;
// //                         }
                        
// //                         if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
// //                             // if(index==344033 and count_f_min==2){std::cout<<"在这里22"<<d<<std::endl;}
// //                             decp_data[next_vertex]= decp_data[index]-diff;
// //                         }
// //                         else if(d1>=1e-16){
// //                             // if(index==344033 and count_f_min==2){std::cout<<"在这里23"<<d<<std::endl;}
// //                             decp_data[next_vertex]-=d1;
// //                         }
// //                         else if(d>=1e-16){
// //                             // if(index==344033 and count_f_min==2){std::cout<<"在这里24"<<d<<std::endl;}
// //                             decp_data[index]+=d;
// //                         }

                    
                    
// //                 }
// //                 else{
// //                     if(decp_data[index]<=decp_data[next_vertex]){
                        
// //                             while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
// //                                     diff/=2;
// //                             }
                            
                            
// //                             if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-16){
// //                                 // while(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff<1e-17){
// //                                 //     diff*=2;
// //                                 // }
// //                                 // if(index==270808 and count_f_min==1){std::cout<<"在这里2！"<< std::endl;}
// //                                 while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
// //                                     diff*=2;
// //                                 }
// //                                 if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
// //                                     decp_data[next_vertex] = decp_data[index]-diff;
// //                                 }
// //                                 // if(index == 6595 and count_f_min==5){
// //                                 //     std::cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;

// //                                 // }
// //                                 // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
// //                                 // decp_data[next_vertex] = decp_data[index]-diff;
// //                                 // if(index==89797){
// //                                 //         std::cout<<"在这里2"<<diff<<", "<<d<<std::endl;
// //                                 // }

// //                                 // decp_data[index]+=d;
// //                             }
// //                             // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
// //                             //     if(index==135569){std::cout<<"在这里23"<<std::endl;}
// //                             //     decp_data[index]+=d;
// //                             // }
// //                             else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
// //                                 while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-17){
// //                                     d1*=2;
// //                                 }
// //                                 // if(count_f_min<=12){std::cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< std::endl;}
// //                                 if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-16){
// //                                     decp_data[next_vertex]-=d1;
// //                                 }
// //                                 // else{
// //                                 //     decp_data[index] += d;
// //                                 // }
// //                                 // else{
// //                                 // decp_data[next_vertex] = input_data[next_vertex] - bound;}
                                
// //                             }
// //                             else{
// //                                 decp_data[next_vertex] = input_data[next_vertex] - bound;
// //                                 //if(index == 6595 and count_f_min==5){std::cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< std::endl;}
// //                             }
                            
                            
                        
                        
// //                 };

// //                 }
                
                

                
// //             }

// //             else{
                
// //                 if(decp_data[index]<decp_data[next_vertex]){
// //                     // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
// //                     //     std::cout<<"np下降："<<std::endl;
// //                     //     std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
// //                     //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
// //                     //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
// //                     //     std::cout<<"diff: "<<diff<<std::endl;
// //                     //     std::cout<<"d: "<<d<<std::endl;
                
// //                     //     }
                        
// //                         // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
// //                         //     double t = decp_data[index];
// //                         //     decp_data[index] = decp_data[next_vertex];
// //                         //     if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
// //                         //     decp_data[next_vertex] = t;
                            
// //                         // }
// //                         if(abs(input_data[next_vertex]-decp_data[index])<bound){
// //                             double t = (decp_data[index]-(input_data[index]-bound))/2.0;
// //                             // if(index==949999){std::cout<<"在这里24"<<std::endl;}
// //                             // decp_data[index] = decp_data[next_vertex];
// //                             // if(next_vertex==66783){std::cout<<"在这里14"<<std::endl;}
// //                             decp_data[next_vertex] = decp_data[index]-t;
                            
// //                         }
// //                         else{
// //                             //if(index==949999){std::cout<<"在这里29"<<std::endl;}
// //                             decp_data[index] = input_data[index] + bound;
// //                         }
// //                 }
                
// //                 else if(decp_data[index]==decp_data[next_vertex]){
// //                     double d = (bound - (input_data[index]-decp_data[index]))/64;
// //                     // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
// //                     //         d/=2;
// //                     // }
// //                     // if(index==949999){
// //                     //     std::cout<<"在这里99 "<<d<<std::endl;
// //                     // }   
// //                     // double d = 1e-16;
// //                     if(abs(input_data[index]-decp_data[index]-d)<=bound){
// //                         decp_data[index]+=d;
// //                     }
// //                     else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
// //                         // if(next_vertex==66783){std::cout<<"在这里13"<<std::endl;}
// //                         decp_data[next_vertex]-=d;
// //                     }
// //                 }
// //             }
            

            
            
            
// //         // if(index == 6595 and count_f_min==5){
// //         //         std::cout<<"下降后："<<std::endl;
// //         //         std::cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<std::endl;
// //         //         std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
// //         //         std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<std::endl;
// //         //         std::cout<<"diff: "<<diff<<std::endl;
// //         //         std::cout<<"d: "<<d<<std::endl;
// //         //         std::cout<<"d1: "<<d1<<std::endl;
// //         //         std::cout<<input_data[index]<<","<<input_data[next_vertex]<<std::endl;
// //         //     }
            
        
// //         }
    
// //         else{
            
// //             int largest_index = from_direction_to_index(index,de_direction_ds[index]);
// //             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
// //             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
// //             // if(count_f_min==84){
// //             //     std::cout<<"np下降："<<std::endl;
// //             //     std::cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<std::endl;
// //             //     std::cout<<"index: "<<index<<", "<<decp_data[index]<<std::endl;
// //             //     std::cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<std::endl;
// //             //     std::cout<<"diff: "<<diff<<std::endl;
// //             //     std::cout<<"d: "<<d<<std::endl;
                
// //             // }
// //             if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
// //                 de_direction_ds[index] = -1;
// //                 return 0;
// //             }
            
// //             if (diff>=1e-16){
// //                 if (decp_data[index]>=decp_data[largest_index]){
// //                     while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
// //                         diff/=2;
// //                     }
                    
                    
// //                     if(abs(input_data[index]-decp_data[index]+diff)<=bound){
// //                         // if(index==999973){
// //                         //     std::cout<<"在这里2！"<<std::endl;
// //                         // }
                        
// //                         decp_data[index] -= diff;
// //                     }
                    
                    
// //                 }                    
// //             }
            
                    
// //             else{
// //                 if (decp_data[index]>=decp_data[largest_index]){
                    
// //                     // if(index==66783){std::cout<<"在这里15"<<std::endl;}
// //                     decp_data[index] = input_data[index] - bound;
// //                 }   
    
// //             }


               
// //         }

        
// //     }    
// //     return 0;
// // };
// // int fix_maxi_critical(int index, int direction){
    
// //     if (direction == 0){
        
// //         if (or_direction_as[index]!=-1){
            
// //             int next_vertex = from_direction_to_index(index,or_direction_as[index]);
// //             int smallest_vertex = next_vertex;
// //             double threshold = std::numeric_limits<double>::lowest();
            
            
// //             for(int i:adjacency[index]){
// //                 if(lowGradientIndices[i] == 1){
// //                     continue;
// //                 }
// //                 if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
// //                     smallest_vertex = i;
// //                     threshold = input_data[i];
// //                 }
// //             }
// //             // if(lowGradientIndices[index] == 1 or lowGradientIndices.find(next_vertex)!=lowGradientIndices.end() or lowGradientIndices.find(smallest_vertex)!=lowGradientIndices.end()){
// //             //         std::cout<<index<<","<<next_vertex<<","<<smallest_vertex<<std::endl;
// //             //     }
// //             threshold = decp_data[smallest_vertex];
// //             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
// //             double d = (bound - (input_data[index]-decp_data[index]))/2.0;
            
// //             if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
// //                 de_direction_as[index]=or_direction_as[index];
            
// //                 return 0;
// //             }
            
// //             if(d>=1e-16){
                
// //                 if(decp_data[index]==decp_data[next_vertex])
// //                     {
                        
// //                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
// //                             d/=2;
// //                         }
// //                         if (abs(input_data[index]-decp_data[index]+d)<=bound){
// //                             decp_data[index] -= d;
// //                         }

                    
// //                     }
// //                 else{
// //                     if(decp_data[index]>=decp_data[next_vertex]){
                        
// //                         while(abs(input_data[index]-decp_data[index]+d)>bound and d>=2e-16){
// //                                 d/=2;
// //                         }
                        
// //                         if(decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
// //                             while(decp_data[index] - d < threshold and d>=2e-16)
// //                             {
// //                                 d/=2;
// //                             }
                            
                            
// //                         }
// //                         else if(threshold>decp_data[next_vertex]){
                            
                            
// //                             double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/2;
                            
// //                             if(diff2>=1e-16){
// //                                 while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
// //                                     diff2/=2;
// //                                 }
                                
// //                                 if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
// //                                     decp_data[smallest_vertex]-=diff2;
// //                                     // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
// //                                 }
                                
                                
// //                             }
                            
// //                         }

// //                         if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex]){
// //                             decp_data[index] -= d;
                            
// //                         }
                        
                        
                   
// //                 };
// //                      }

                 
            
                
// //             }
// //             else{
                
// //                 if(decp_data[index]>=decp_data[next_vertex]){
// //                     if(abs(input_data[index]-(input_data[next_vertex] -bound+ decp_data[index])/2.0)<=bound){
// //                         decp_data[index] = (input_data[next_vertex] -bound + decp_data[index])/2.0;
// //                     }
// //                     else{
                        
// //                         decp_data[index] = input_data[index] - bound;
// //                     }
                    
// //                 }
                
// //             }
            
            
        
// //         }
// //         else{
            
// //             int largest_index = from_direction_to_index(index,de_direction_as[index]);
// //             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
// //             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
// //             if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
// //                 de_direction_as[index] = -1;
// //             }
// //             if(d>=1e-16){
                
// //                 if (decp_data[index]<=decp_data[largest_index]){
// //                     if(abs(input_data[largest_index]-decp_data[index]+d)){
// //                         decp_data[largest_index] = decp_data[index]-d;
// //                     }
// //                 }
                
            
                
// //             }
            
// //             else{
// //                 if(decp_data[index]<=decp_data[largest_index]){
// //                     decp_data[index] = input_data[index] + bound;
// //                 }
                    
// //             }
            
// //         }
        
        
    
// //     }
    
// //     else{
        
// //         if (or_direction_ds[index]!=-1){
// //             int next_vertex= from_direction_to_index(index,or_direction_ds[index]);
            
// //             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
// //             double d =  (bound-(input_data[index]-decp_data[index]))/2.0;
            
            
// //             if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
// //                 de_direction_ds[index]=or_direction_ds[index];
// //                 return 0;
// //             }
            
// //             if(diff>=1e-16 or d>=1e-16){
// //                 if(decp_data[index]==decp_data[next_vertex]){
                    
                    
// //                         while(abs(input_data[next_vertex]-decp_data[index]-d)>bound and d>=2e-16){
// //                             d/=2;
// //                         }
                        
// //                         if(abs(input_data[index]-decp_data[index]-d)<=bound){
// //                             decp_data[index]+=d;
// //                         }
                    
                    
                    
                    
// //                 }
// //                 else{
// //                     if(decp_data[index]<=decp_data[next_vertex]){
                        
// //                             while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
// //                                     diff/=2;
// //                             }
                            
// //                             if (abs(input_data[next_vertex]-decp_data[index]+d)<=bound and decp_data[index]<=decp_data[next_vertex]){
// //                                 decp_data[next_vertex] = decp_data[index]-diff;
// //                             }
                            
                            
                        
                        
// //                 };

// //                 }
                
                

                
// //             }

// //             else{
                
// //                 if(decp_data[index]<=decp_data[next_vertex]){
// //                     if(abs(input_data[index]-(input_data[next_vertex] + bound + decp_data[index])/2.0)<=bound){
// //                         decp_data[index] = (input_data[next_vertex] + bound + decp_data[index])/2.0;
// //                     }
// //                     else{
// //                         decp_data[index] = input_data[index] + bound;
// //                     }
// //                 }
// //             }
            

            
            
            

            
        
// //         }
    
// //         else{
            
// //             int largest_index = from_direction_to_index(index,de_direction_ds[index]);
// //             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
// //             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            
// //             if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
// //                 de_direction_ds[index] = -1;
// //                 return 0;
// //             }
            
// //             if (diff>=1e-16){
// //                 if (decp_data[index]>=decp_data[largest_index]){
// //                     while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
// //                         diff/=2;
// //                     }
                    
                    
// //                     if(abs(input_data[index]-decp_data[index]+diff)<=bound){
// //                         decp_data[index] -= diff;
// //                     }
                    
                    
// //                 }                    
// //             }
            
                    
// //             else{
// //                 if (decp_data[index]>=decp_data[largest_index]){
// //                     decp_data[index] = input_data[index] - bound;
// //                 }   
    
// //             }


               
// //         }

        
// //     }    
// //     return 0;
// // };
// // int fixpath1(int index, int direction){
    
// //     if(direction==0){
// //         int cur = index;
        
// //         while (or_direction_as[cur] == de_direction_as[cur]){
// //             int next_vertex =  from_direction_to_index(cur,de_direction_as[cur]);
            
// //             if(de_direction_as[cur]==-1 && next_vertex == cur){
// //                 cur = -1;
// //                 break;
// //             }
// //             if(next_vertex == cur){
// //                 cur = next_vertex;
// //                 break;
// //             };
            
// //             cur = next_vertex;
// //         }

// //         int start_vertex = cur;
        
        
// //         if (start_vertex==-1) return 0;
// //         else{
            
// //             int false_index= from_direction_to_index(cur,de_direction_as[cur]);
// //             int true_index= from_direction_to_index(cur, or_direction_as[cur]);
// //             if(false_index==true_index) return 0;
            
// //             // double diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
// //             double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
// //             // double d = (bound-abs(input_data[true_index]-decp_data[true_index]))/2.0;
// //             double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            
// //             // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            
// //             if(decp_data[false_index]<decp_data[true_index]){
// //                 de_direction_as[cur]=or_direction_as[cur];
// //                 // 103781, 103830
// //                 return 0;
// //             }
// //             double threshold = std::numeric_limits<double>::lowest();
// //             int smallest_vertex = false_index;
            
// //             for(int i:adjacency[false_index]){
// //                 if(lowGradientIndices[i] == 1){
// //                 continue;}
// //                 if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
// //                     smallest_vertex = i;
// //                     threshold = input_data[i];
// //                 }
// //             }
            
// //             threshold = decp_data[smallest_vertex];

// //             double threshold1 = std::numeric_limits<double>::max();
// //             int smallest_vertex1 = true_index;
            
// //             for(int i:adjacency[true_index]){
// //                 if(lowGradientIndices[i] == 1){
// //                 continue;}
// //                 if(input_data[i]>input_data[true_index] and input_data[i]<threshold1 and i!=true_index){
// //                     smallest_vertex1 = i;
// //                     threshold = input_data[i];
// //                 }
// //             }
            
// //             threshold1 = decp_data[smallest_vertex1];

// //             if (diff>=1e-16 or d>=1e-16){
// //                 if (decp_data[false_index]>=decp_data[true_index]){

                    
// //                     // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
// //                     while(abs(input_data[false_index]-decp_data[false_index] + d)>bound and d>2e-16){
// //                                 d/=2;
// //                     }
                    
                    
// //                     if(decp_data[false_index]>threshold and threshold<decp_data[true_index]){
                            
// //                             while(decp_data[false_index] - d < threshold and d>=2e-16)
// //                             {
// //                                 d/=2;
// //                             }
                            
                            
// //                     }
// //                     else if(threshold>=decp_data[true_index]){
                        
                        
// //                         double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/64;
                        
// //                         if(diff2>=1e-16){
// //                             while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[true_index]){
                                
// //                                 diff2/=2;
// //                             }
                            
// //                             if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
// //                                 decp_data[smallest_vertex]-=diff2;
// //                                 // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
// //                             }
                            
                            
// //                         }
                        
// //                     }
// //                     while(abs(input_data[true_index]-(decp_data[false_index] + diff))>bound and diff>2e-16){
// //                                 diff/=2;
// //                     }
// //                     if(decp_data[true_index]<=threshold and threshold>=decp_data[false_index]){
                            
// //                             while(decp_data[false_index] + diff > threshold and diff>=2e-16)
// //                             {
// //                                 diff/=2;
// //                             }
                            
                            
// //                     }
// //                     // else if(threshold<=decp_data[false_index]){
                        
                        
// //                     //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/64;
                        
// //                     //     if(diff2>=1e-16){
// //                     //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]+diff2<decp_data[false_index]){
                                
// //                     //             diff2/=2;
// //                     //         }
                            
// //                     //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)<=bound){
// //                     //             decp_data[smallest_vertex]+=diff2;
// //                     //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
// //                     //         }
                            
                            
// //                     //     }
                        
// //                     // }
// //                     if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index] and diff>=1e-16){
// //                         decp_data[true_index] = decp_data[false_index] + diff;
// //                     }
// //                     else if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound and d>=1e-16){
// //                         decp_data[false_index] -=d;
// //                     }
                    
                    
                        
// //                 }

// //                 else{
// //                     de_direction_as[cur] = or_direction_as[cur];
// //                 }
                    
// //             }
            
// //             else{
// //                 double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
// //                 if (decp_data[false_index]>=decp_data[true_index]){
// //                     if(abs(input_data[false_index]-((decp_data[false_index]+input_data[true_index]-bound)/2.0))<=bound){
// //                         decp_data[false_index] = (decp_data[false_index]+input_data[true_index]-bound)/2.0;
// //                     }
                        
// //                     else{
// //                         decp_data[false_index] = input_data[false_index] - bound;
// //                     }
                    
// //                 }
// //                 else{
// //                     de_direction_as[cur] = or_direction_as[cur];
// //                 };        
// //             }
            
// //         }
// //     }

// //     else{
// //         int cur = index;
        
        
// //         while (or_direction_ds[cur] == de_direction_ds[cur]){
            
// //             int next_vertex = from_direction_to_index(cur,de_direction_ds[cur]);
            
// //             if(next_vertex == cur){
// //                 cur = -1;
// //                 break;
// //             }
// //             if (next_vertex == cur){
// //                 cur = next_vertex;
// //                 break;
// //             }
// //             cur = next_vertex;

// //             if (cur == -1) break;
                
// //         }
    
// //         int start_vertex = cur;
        
        
// //         if (start_vertex==-1) return 0;
        
// //         else{
            
// //             int false_index= from_direction_to_index(cur,de_direction_ds[cur]);
// //             int true_index= from_direction_to_index(cur, or_direction_ds[cur]);
// //             if(false_index==true_index) return 0;
            
// //             // double diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
// //             double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
// //             // double d = (bound-abs(input_data[true_index]-decp_data[true_index]))/2.0;
// //             double d = (bound+input_data[false_index]-decp_data[false_index])/2.0;
            
// //             // diff是对true_index做减法，
// //             // d是对false_index做加法
// //             // if(decp_data[false_index]>decp_data[true_index]){
                
// //             //     de_direction_ds[cur]=or_direction_ds[cur];
            
// //             //     return 0;
// //             // }
            
              
// //             if(diff>=1e-16 or d>=1e-16){
// //                 if(decp_data[false_index]<=decp_data[true_index]){
                    
// //                     // else{
                        
// //                         // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
// //                         while(abs(input_data[false_index]-decp_data[false_index] - d)>bound and d>=2e-17){
// //                             d/=2;
// //                         }
// //                         while(abs(input_data[true_index]-(decp_data[false_index] - diff))>bound and diff>=2e-17){
// //                                     diff/=2;
// //                         }
// //                         if(abs(input_data[true_index]-(decp_data[false_index] - diff))<=bound and decp_data[false_index]<=decp_data[true_index]){
// //                             // decp_data[false_index] = decp_data[true_index] + diff;
// //                             decp_data[true_index] = decp_data[false_index] - diff;
// //                         }
// //                         if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
// //                             decp_data[false_index] += d;
// //                         }
                        
                        
                        

// //                         // diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
                        
                        
                        
// //                         // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
// //                         if (decp_data[false_index]==decp_data[true_index]){
// //                             if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                        
// //                                 decp_data[false_index] += d;
// //                         }
                       
// //                         }
// //                     // }
                    
// //                     }
                
// //                     else{
// //                         de_direction_ds[cur] = or_direction_ds[cur];
// //                     }
// //             }

// //             else{
                
// //                 double diff = (bound-(input_data[false_index]-decp_data[false_index]))/100000.0;
                
// //                 if(decp_data[false_index]<=decp_data[true_index]){
// //                     if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/100000.0))<=bound){
// //                         decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/100000.0;
// //                     }
// //                     else{
// //                         decp_data[false_index] = input_data[false_index] + bound;
// //                     }
// //                     while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound){
// //                         diff/=2;
// //                     }
// //                     if (decp_data[false_index]==decp_data[true_index]){
// //                         double diff = (bound-(input_data[false_index]-decp_data[false_index]))/100000.0;
// //                         decp_data[false_index]+=diff;
// //                     }
                
// //                 }
            
// //                     else{
// //                         de_direction_ds[cur] = or_direction_ds[cur];
// //                     }
// //             }
// //         }
// //     }

// //     return 0;
// // };
// int fixpath(int index, int direction){
//     if(direction==0){
//         int cur = index;
//         while (or_direction_as[cur] == de_direction_as[cur]){
//             int next_vertex =  from_direction_to_index(cur,de_direction_as[cur]);
            
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
            
//             int false_index= from_direction_to_index(cur,de_direction_as[cur]);
//             int true_index= from_direction_to_index(cur, or_direction_as[cur]);
//             if(false_index==true_index) return 0;
//             // 对的
//             double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
//             // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
//             // 对的
//             double d = (decp_data[false_index]-input_data[false_index]+bound)/2.0;
//             // double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
//             // diff是用来给true_index增加的
//             // d是用来给false_index见效的
//             // double diff = (input_data[true_index]-bound_data[true_index]-decp_data[false_index])/2.0;
//             // double d = (input_data[false_index]-bound_data[false_index]-decp_data[false_index])/2.0;
//             // if(wrong_index_as.size()<=50){
//             // // pre=1;
//             //     std::cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<std::endl;
//             //     std::cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<std::endl;
//             //     std::cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<std::endl;
//             //     std::cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<std::endl;
//             //     std::cout<<diff<<std::endl;
//             //     std::cout<<d<<std::endl;
//             // }
//             if(decp_data[false_index]<decp_data[true_index]){
//                 de_direction_as[cur]=or_direction_as[cur];
//             //     if(wrong_maxi_cp.size()==1 and wrong_min_cp.size()==0){
//             //     std::cout<<de_direction_as[64582]<<std::endl;
//             // }
//                 return 0;
//             }
//             double threshold = std::numeric_limits<double>::lowest();
//             int smallest_vertex = false_index;
            
//             for(int i:adjacency[false_index]){
                
//                 if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
//                     smallest_vertex = i;
//                     threshold = input_data[i];
//                 }
//             }
            
//             threshold = decp_data[smallest_vertex];

//             double threshold1 = std::numeric_limits<double>::max();
//             int smallest_vertex1 = true_index;
            
//             for(int i:adjacency[true_index]){
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
                        
                        
//                         double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
//                         if(diff2>1e-16){
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
                        
                        
//                     //     double diff2 = (bound-(input_data[smallest_vertex1]-decp_data[smallest_vertex1]))/2;
                        
//                     //     if(diff2>1e-16){
//                     //         while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]+diff2<decp_data[false_index]){
                                
//                     //             diff2/=2;
//                     //         }
                            
//                     //         if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]-diff2)<=bound){
//                     //             decp_data[smallest_vertex]+=diff2;
//                     //             // if(index==97) std::cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<std::endl;
//                     //         }
                            
                            
//                     //     }
                        
//                     // }
//                     if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index]){
//                         // ascending version;
//                         decp_data[true_index] = decp_data[false_index] + diff;
//                         // decp_data[false_index] = decp_data[true_index]-diff;
//                     }
//                     if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
//                         // decp_data[true_index] +=d;
//                         decp_data[false_index] -=d;
//                     }
                    
                    
                        
//                 }

//                 else{
//                     de_direction_as[cur] = or_direction_as[cur];
//                 }
                    
//             }
            
//             else{
//                 //对的
//                 double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
//                 // if(wrong_index_as.size()==2){
//                 //     std::cout<<diff<<std::endl;
//                 //     std::cout<<false_index<<std::endl;
//                 // }
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
            
//             int next_vertex = from_direction_to_index(cur,de_direction_ds[cur]);
            
//             // if(de_direction_ds[cur]==-1 && next_vertex == cur){
//             //     if(wrong_index_ds.size()==4){
//             //         std::cout<<cur<<", "<<index <<", "<<de_direction_ds[cur]<<", "<<or_direction_ds[cur]<<std::endl;
//             //     }
//             //     cur = -1;
//             //     break;
//             // }
//             if (next_vertex == cur){
//                 cur = next_vertex;
//                 break;
//             }
//             cur = next_vertex;

//             // if (cur == -1) break;
                
//         }
    
//         int start_vertex = cur;
//         // if(wrong_index_ds.size()==4){
//         //     std::cout<<"修复的时候变成了:" <<std::endl;
//         //     std::cout<<start_vertex<<", "<<de_direction_ds[start_vertex]<<", "<<or_direction_ds[start_vertex]<<std::endl;
//         // }
//         if (start_vertex==-1) return 0;
        
//         else{
            
//             int false_index= from_direction_to_index(cur,de_direction_ds[cur]);
//             int true_index= from_direction_to_index(cur, or_direction_ds[cur]);
//             if(false_index==true_index) return 0;

//             // double diff = (input_data[true_index]+bound-decp_data[false_index])/2.0;
//             double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
//             // double d = (input_data[false_index]bound-decp_data[false_index])/1000.0;
//             // double d = (input_data[false_index]+bound-decp_data[false_index])/2.0;
//             // double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
//             // // double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
//             // double d = (input_data[false_index]-bound-decp_data[false_index])/2.0;
//             double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
//             // diff是用来给true_index增加的
//             // d是用来给false_index见效的
//             // if(wrong_index_as.size()<=10){
//             //     std::cout<<index<<", "<<decp_data[index]<<"," <<input_data[index]<<std::endl;
//             //     std::cout<<start_vertex<<", "<<decp_data[start_vertex]<<"," <<input_data[start_vertex]<<std::endl;
//             //     std::cout<<false_index<<", "<<decp_data[false_index]<<"," <<input_data[false_index]<<std::endl;   
//             //     std::cout<<true_index<<", "<<decp_data[true_index]<<"," <<input_data[true_index]<<std::endl;                     
//             // }
//             if(decp_data[false_index]>decp_data[true_index]){
//                 de_direction_ds[cur]=or_direction_ds[cur];
//                 return 0;
//             }
            
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
                       
//                     }
//                     // }
                    
//                 }
            
//                 else{
//                     de_direction_ds[cur] = or_direction_ds[cur];
//                 }
//             }

//             else{
                
//                 double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                
//                 if(decp_data[false_index]<=decp_data[true_index]){
//                     if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/2.0))<=bound){
//                         decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/2.0;
//                     }
//                     else{
//                         decp_data[false_index] = input_data[false_index] + bound;
//                     }
//                     while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound and diff>=2e-17){
//                         diff/=2;
//                     }
//                     if (decp_data[false_index]==decp_data[true_index]){
//                         double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
//                         decp_data[false_index]+=diff;
//                     }
                
//                 }
            
//                 else{
//                     de_direction_ds[cur] = or_direction_ds[cur];
//                 }
//             }
//         }
//     }

//     return 0;
// };
// // int get_wrong_index_cp(){
    
// //     return 0;
// // };
// double calculateMSE(const std::vector<double>& original, const std::vector<double>& compressed) {
//     if (original.size() != compressed.size()) {
//         throw std::invalid_argument("The size of the two vectors must be the same.");
//     }

//     double mse = 0.0;
//     for (size_t i = 0; i < original.size(); i++) {
//         mse += std::pow(static_cast<double>(original[i]) - compressed[i], 2);
//     }
//     mse /= original.size();
//     return mse;
// }

// double calculatePSNR(const std::vector<double>& original, const std::vector<double>& compressed, double maxValue) {
//     double mse = calculateMSE(original, compressed);
//     if (mse == 0) {
//         return std::numeric_limits<double>::infinity(); // Perfect match
//     }
//     double psnr = -20.0*log10(sqrt(mse)/maxValue);
//     return psnr;
// }

// double get_wrong_index_path(){
    
//     wrong_index_as.clear();
//     wrong_index_ds.clear();
//     double cnt = 0.0;
//     for (int index = 0; index < size2; ++index) {
//         if(lowGradientIndices[index] == 1){
//             continue;
//         }
//         if(or_label[index*2+1] != dec_label[index*2+1] || or_label[index*2] != dec_label[index*2]){
//             cnt+=1.0;
//         }
//         if (or_label[index*2+1] != dec_label[index*2+1]) {
//             wrong_index_as.push_back(index);
//         }
//         if (or_label[index*2] != dec_label[index*2]) {
//             wrong_index_ds.push_back(index);
//         }
// }
//     // std::cout<<cnt/size2<<std::endl;
//     double result = static_cast<double>(cnt) / static_cast<double>(size2);
//     return result;
// };
// // bool compareIndices(int i1, int i2, const std::vector<double>& data) {
// //     if (data[i1] == data[i2]) {
// //         return i1<i2; // 注意这里是‘>’
// //     }
    
// //     return data[i1] < data[i2];
// // }

// // void get_false_criticle_points(){
// //     count_max=0;
// //     count_min=0;

// //     count_f_max=0;
// //     count_f_min=0;

// //     #pragma omp parallel for
// //     for (auto i = 0; i < size2; i ++) {
// //             if(lowGradientIndices[i] == 1){
// //                 continue;
// //             }
// //             bool is_maxima = true;
// //             bool is_minima = true;

// //             for (int j : adjacency[i]) {
// //                 if(lowGradientIndices[j] == 1){
// //                     continue;
// //                 }
// //                 if (decp_data[j] > decp_data[i]) {
// //                     is_maxima = false;
// //                     break;
// //                 }
// //                 else if(decp_data[j] == decp_data[i] and j>i){
// //                     is_maxima = false;
// //                     break;
// //                 }
// //             }
// //             for (int j : adjacency[i]) {
// //                 if(lowGradientIndices[j] == 1){
// //                     continue;
// //                 }
// //                 if (decp_data[j] < decp_data[i]) {
// //                     is_minima = false;
// //                     break;
// //                 }
// //                 else if(decp_data[j] == decp_data[i] and j<i){
// //                     is_minima = false;
// //                     break;
// //                 }
// //             }
        
// //     if(((is_maxima && (or_direction_as[i]!=-1)) or (!is_maxima && (or_direction_as[i]==-1)))){
// //         // if(lowGradientIndices[i] == 0){
// //         //     std::cout<<or_direction_as[i]<<de_direction_as[i]<<std::endl;
// //         // }
// //         // if(lowGradientIndices[i] == 0){
// //             int idx_fp_max = atomic_fetch_add_explicit(&count_f_max, 1, memory_order_relaxed);
// //             // if(i==25026){std::cout<<or_direction_as[i]<<is_maxima<<std::endl;}
// //             all_max[idx_fp_max] = i;
// //         // }
        
// //     }

// //     else if (((is_minima && (or_direction_ds[i]!=-1)) or (!is_minima && or_direction_ds[i]==-1)) and lowGradientIndices[i] == 0) {
// //         int idx_fp_min = atomic_fetch_add_explicit(&count_f_min, 1, memory_order_relaxed);// in one instruction
// //         all_min[idx_fp_min] = i;
    
// //     }     
        
// // }
    

// // }
// // extern "C" {
// //     int mainCuda(std::vector<int> a, std::vector<int> b);
// // }
// // fix_process
// // extern __global__ void init_or_data1(int numElements);
// // extern void init_or_data(std::vector<int> *a, std::vector<int> *b,std::vector<int> *c, std::vector<int> *d, std::vector<double> *input_data,std::vector<double> *decp_data, int num);
// // extern int fix_process(std::vector<double>& decp_data);
// // std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,int width1, int height1, int depth1, std::vector<int> *low,double bound1,float &datatransfer,float &finddirection
// extern void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,std::vector<int>* dec_label1,std::vector<int>* or_label1,int width1, int height1, int depth1, std::vector<int> *low,double bound1,float &datatransfer,float &finddirection);
// // extern void update_de_direction(std::vector<int> *c,std::vector<int> *d);
// extern void fix_process(std::vector<int> *c,std::vector<int> *d, std::vector<double> *decp_data1, float &datatransfer, float &finddirection, float &getfcp, float &fixtime_cp,int &cpite);
// extern void mappath1(std::vector<int> *label, std::vector<int> *direction_as, std::vector<int> *direction_ds, float &finddirection, float &mappath_path, float &datatransfer,int type=0);

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
// double maxAbsoluteDifference(const std::vector<double>& vec1, const std::vector<double>& vec2) {
//     if (vec1.size() != vec2.size()) {
//         std::cerr << "Vectors are of unequal size." << std::endl;
//         return -1; // Or handle the error as per your need
//     }

//     double maxDiff = 0.0;
//     for (size_t i = 0; i < vec1.size(); ++i) {
//         double diff = std::abs(vec1[i] - vec2[i]);
//         if (diff < maxDiff) {
//             maxDiff = diff;
//         }
//     }

//     return maxDiff;
// }
// int main(int argc, char** argv){
//     omp_set_num_threads(44);
//     std::string dimension = argv[1];
//     double range = std::stod(argv[2]);
//     std::string compressor_id = argv[3];
//     int mode = std::stoi(argv[4]);
//     double target_br;
//     float datatransfer = 0.0;
//     float mappath_path = 0.0;
//     float getfpath = 0.0;
//     float fixtime_path = 0.0;
//     float finddirection = 0.0;
//     float getfcp = 0.0;
//     float fixtime_cp = 0.0;
//     if(mode==1){
//         target_br = std::stod(argv[5]);
//     }
//     std::istringstream iss(dimension);
//     char delimiter;
//     std::string filename;
//     if (std::getline(iss, filename, ',')) {
//         // 接下来读取整数值
//         if (iss >> width >> delimiter && delimiter == ',' &&
//             iss >> height >> delimiter && delimiter == ',' &&
//             iss >> depth) {
//             std::cout << "Filename: " << filename << std::endl;
//             std::cout << "Width: " << width << std::endl;
//             std::cout << "Height: " << height << std::endl;
//             std::cout << "Depth: " << depth << std::endl;
//         } else {
//             std::cerr << "Parsing error for dimensions" << std::endl;
//         }
//     } else {
//         std::cerr << "Parsing error for filename" << std::endl;
//     }

    
//     inputfilename = "experiment_data/"+filename+".bin";
    
    

    
    
//     // 显式指定模板参数为 double
//     // std::vector<double> dataDouble = getdata2<double>("datafile.txt");
//     auto start = std::chrono::high_resolution_clock::now();
//     input_data = getdata2(inputfilename);
//     auto min_it = std::min_element(input_data.begin(), input_data.end());
//     auto max_it = std::max_element(input_data.begin(), input_data.end());
//     double minValue = *min_it;
//     double maxValue = *max_it;
//     bound = (maxValue-minValue)*range;
    
//     // exit(0);
//     std::ostringstream stream;
//     std::cout.precision(std::numeric_limits<double>::max_digits10);
//     // 设置精度为最大可能的双精度浮点数有效数字
//     stream << std::setprecision(std::numeric_limits<double>::max_digits10);
//     stream << std::defaultfloat << bound;  // 使用默认的浮点表示
//     std::string valueStr = stream.str();
//     size2 = input_data.size();
//     std::string cpfilename = "./compressed_data/compressed_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".sz";
//     std::string decpfilename = "./decompressed_data/decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
//     std::string fix_path = "./decompressed_data/fixed_decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
//     std::string command;
//     std::cout<<decpfilename<<std::endl;
//     std::cout<<bound<<", "<<std::to_string(bound)<<std::endl;
//     // exit(0);
//     int result;
    
//     if(compressor_id=="sz3"){
        
//         command = "sz3 -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -M "+"ABS "+std::to_string(bound)+" -a";
//         std::cout << "Executing command: " << command << std::endl;
//         result = std::system(command.c_str());
//         if (result == 0) {
//             std::cout << "Compression successful." << std::endl;
//         } else {
//             std::cout << "Compression failed." << std::endl;
//         }
//     }
//     else if(compressor_id=="zfp"){
//         cpfilename = "compressed_data/compressed_"+filename+"_"+std::to_string(bound)+".zfp";
//         // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
//         command = "zfp -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -a "+std::to_string(bound)+" -s";
//         std::cout << "Executing command: " << command << std::endl;
//         result = std::system(command.c_str());
//         if (result == 0) {
//             std::cout << "Compression successful." << std::endl;
//         } else {
//             std::cout << "Compression failed." << std::endl;
//         }
//     }


    
    
//     decp_data = getdata2(decpfilename);
//     auto ends = std::chrono::high_resolution_clock::now();
    
//     std::chrono::duration<double> duration = ends - start;
//     double compression_time = duration.count();
//     std::cout<<"before_gpu: "<<decp_data[96955]<<std::endl;
//     std::vector<double> decp_data_copy(decp_data);
    
//     or_direction_as.resize(size2);
//     or_direction_ds.resize(size2);
//     de_direction_as.resize(size2);
//     de_direction_ds.resize(size2);
//     or_label.resize(size2*2, -1);
//     dec_label.resize(size2*2, -1);
//     // std::vector<std::vector<double>> time_counter;
//     lowGradientIndices = find_low();
//     // lowGradientIndices.resize(size2, 0);
    
//     adjacency = _compute_adjacency();
    
//     all_max.resize(size2);
//     all_min.resize(size2);
    
//     std::random_device rd;  // 随机数生成器
//     std::mt19937 gen(rd()); // 以随机数种子初始化生成器
    
    
    

//     int cnt=0;
 
    
    
    
    
    
    
    
//     // #pragma omp parallel for
//     // for (int i=0;i<size2;++i){
//     //     // if(i%100000==0) std::cout<<i/100000<<std::endl;
//     //     if(lowGradientIndices[i] == 1){
//     //         continue;
//     //     }
//     //     // if (or_mesh.maxi.find(i) == or_mesh.maxi.end()){
            
//     //         or_direction_as[i] = find_direction(i,input_data,0);
//     //     // }
//     //     // else{
            
//     //     //     or_direction_as[i]= -1;
//     //     // };
        
        
//     //     // if (or_mesh.mini.find(i) == or_mesh.mini.end()){
            
//     //         or_direction_ds[i] = find_direction(i,input_data,1);
            
//     //     // }
//     //     // else{
            
//     //     //     or_direction_ds[i] = -1;
//     //     // };
        
        
       
        

//     //  };
//     // // exit(0);
//     // // std::cout<<"dahk"<<std::endl;

    
    
    
//     // #pragma omp parallel for
    
//     // for (int i=0;i<size2;++i){
//     //             if(lowGradientIndices[i] == 1){
//     //             continue;
//     //         }
//     //         de_direction_as[i] = find_direction(i,decp_data,0);
        
//     //         de_direction_ds[i] = find_direction(i,decp_data,1);
        
//     // };
//     // std::cout<<"de完成"<<std::endl;
    
    


    
//     // or_label = mappath(or_mesh,or_direction_as,or_direction_ds);
//     // std::cout<<lowGradientIndices.size()<<std::endl;
//     // std::cout<<"map完成"<<std::endl;
//     // exit(0);
//     // std::ofstream file1("or"+std::to_string(width)+"_label.txt");
//     // std::string result = "";
//     // for (const auto& p : or_label) {
//     //     result+="[";
//     //     result+=std::to_string(p.first)+","+std::to_string(p.second);
//     //     result+="],";
//     // }
//     // file1 << result << "\n";
//     // file1 << "#"<<",";
    
//     // result = "[";
//     // for( int i:or_mesh.maxi){
//     //     result+=std::to_string(i);
//     //     result+=",";
//     // }
//     // result+="]";
//     // file1 << result;
//     // file1 << "\n";
//     // file1 << "#"<<",";
//     // result = "[";
//     // for( int i:or_mesh.mini){
//     //     result+=std::to_string(i);
//     //     result+=",";
//     // }
//     // result+="]";
//     // file1 << result;

//     // // // 关闭文件
//     // file1.close();
    
//     // std::ofstream file1("or_label.txt");

//     // // 写入数据
    
//     // for (const auto& p : or_label) {
        
//     //     file1 << p.first << ", " << p.second << "\n";
//     // }
    
    
//     // file1 << "#"<<",";
//     // for( int i:or_mesh.maxi){
//     //     file1 << i << ",";
//     // }
//     // file1 << "\n";
//     // file1 << "#"<<",";
//     // for( int i:or_mesh.mini){
//     //     file1 << i << ",";
//     // }
//     // // 关闭文件
//     // file1.close();
    
    
    
    
//     // dec_label = mappath(de_mesh,de_direction_as,de_direction_ds);
    
    
    
    
    
    
    
//     // while (count_f_max>0 or count_f_min>0){
//     //     cpite +=1;
//     //     start5 = std::chrono::high_resolution_clock::now();
        
//     //     std::cout<<"修复cp: "<<count_f_max<<", "<<count_f_min<<std::endl;
//     //     // std::cout<<"修复:"<<or_direction_as[25026]<<", "<<de_direction_as[25026]<<std::endl;
//     //     // if(count_f_max ==3){
//     //     //     for(int i=0;i<count_f_max;i++){
//     //     //         std::cout<<all_max[i]<<std::endl;
                
//     //     //         for(int i:adjacency[i]){
//     //     //             std::cout<<i<<", "<<input_data[i]<<std::endl;
//     //     //         }
//     //     //     }
//     //     //     std::cout<<std::endl;
//     //     // }
//     //     // if(count_f_min == 84){
//     //     //     for(int i=0;i<count_f_min;i++){
//     //     //         std::cout<<all_min[i]<<"," ;
//     //     //     }
//     //     //     std::cout<<std::endl;
//     //     // }
        
//     //     start4 = std::chrono::high_resolution_clock::now();
//     //     #pragma omp parallel for

//     //     for(auto i = 0; i < count_f_max; i ++){
            
//     //         int critical_i = all_max[i];
//     //         fix_maxi_critical(critical_i,0);

//     //     }
        
        
        
//     //     #pragma omp parallel for
//     //     for(auto i = 0; i < count_f_min; i ++){

//     //         int critical_i = all_min[i];
//     //         fix_maxi_critical(critical_i,1);

//     //     }
        
        
       
        
//     //     end4 = std::chrono::high_resolution_clock::now();
//     //     d = end4-start4;
//     //     fixtime_cp += d.count();
        
//     //     start4 = std::chrono::high_resolution_clock::now();
//     //     #pragma omp parallel for
//     //     for (int i=0;i<size2;++i){
            
//     //             if(lowGradientIndices[i] == 1){
//     //                 continue;
//     //             }
//     //             de_direction_as[i] = find_direction(i,decp_data,0);
            
//     //             de_direction_ds[i] = find_direction(i,decp_data,1);
            
            
//     //     };
//     //     // get_wrong_index_cp();
//     //     get_false_criticle_points();
//     //     end4 = std::chrono::high_resolution_clock::now();
//     //     d = end4-start4;
//     //     getfcp += d.count();
        

//     //     d = end4-start5;
//     //     total+= d.count();
        
//     //     // std::cout<<"time for fix:"<<temp<<std::endl;
    
//     //     // std::cout<<"time for assign value:"<<temp2<<std::endl;
//     //     // std::cout<<"time for get cp:"<<temp3<<std::endl;
//     //     // std::cout<<"time for get false cp:"<<temp5<<std::endl;
//     //     // std::cout<<"total time："<<total<<std::endl;
//     //     // std::cout<<"time for fix"<<temp/total<<std::endl;
        
//     //     // std::cout<<"assign value"<<temp2/total<<std::endl;
//     //     // std::cout<<"get cp"<<temp3/total<<std::endl;
//     //     // std::cout<<"get direction"<<temp1/total<<std::endl;
//     //     // std::cout<<"get false cp"<<temp5/total<<std::endl;
//     //     // std::cout<<"错误cp:"<<count_max<<","<<count_min<<std::endl;
//     //     // std::cout<<cpite<<std::endl;
//     //     // exit(0);
//     // }
    
//     // std::cout<<"开始"<<std::endl;
//     // init_or_data(&or_direction_as, &or_direction_ds, &de_direction_as, &de_direction_ds, &input_data, &decp_data,size2);
//     std::vector<int>* dev_a = &or_direction_as;
//     std::vector<int>* dev_b = &or_direction_ds;
//     std::vector<int>* dev_c = &de_direction_as;
//     std::vector<int>* dev_d = &de_direction_ds;
//     std::vector<double>* dev_e = &input_data;
//     std::vector<double>* dev_f = &decp_data;
//     std::vector<int>* dev_g = &lowGradientIndices;
    
//     // init_or_data(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f,size2);
//     std::vector<int>* dev_q = &dec_label;
//     std::vector<int>* dev_m = &or_label;
//     // std::cout<<"开始"<<std::endl;
    
    
    
//     double compressed_dataSize = fs::file_size(cpfilename);
//     double br = (compressed_dataSize*8)/size2;
//     std::vector<std::vector<double>> time_counter;
//     start = std::chrono::high_resolution_clock::now();
//     init_inputdata(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, width, height, depth, dev_g, bound,datatransfer,finddirection);
//     ends = std::chrono::high_resolution_clock::now();
//     duration = ends - start;
//     double additional_time = duration.count();
//     std::cout<<"compression_time: "<<compression_time<<std::endl;
//     std::cout<<"additional_time: "<<additional_time<<std::endl;
//     std::ofstream outFilep("result/compression_time.txt", std::ios::app);
//     // 检查文件是否成功打开
//     if (!outFilep) {
//         std::cerr << "Unable to open file for writing." << std::endl;
//         return 1; // 返回错误码
//     }

//     outFilep << filename + "_" +compressor_id+"_" + std::to_string(bound) + ":" << "compression time:" <<compression_time<<" additional time:" <<additional_time << std::endl;
//     // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
//     // outFilep << std::to_string(number_of_thread)<<":" << std::endl;
    
//     outFilep << "\n"<< std::endl;
//     outFilep.close();
//     return 0;
//     std::cout<<"duration: "<<duration.count()<<std::endl;
//     auto endt = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> durationt = endt - start;
//     std::cout<<"duration: "<<durationt.count()<<std::endl;
//     std::cout<<"find_direction:"<<finddirection*1000<<std::endl;
//     std::cout<<"getfcp:"<<getfcp<<std::endl;
//     std::cout<<"mappath_path:"<<mappath_path<<std::endl;
//     std::cout<<"getfpath:"<<getfpath<<std::endl;
//     std::cout<<"fixfcp:"<<fixtime_cp<<std::endl;
//     std::cout<<"fixpath:"<<fixtime_path<<std::endl;
//     exit(0);
    
    
//     // std::cout<<"wancheng"<<std::endl;
//     // update_de_direction(dev_c,dev_d);
//     // for(int i=0;i<input_data.size();i++){
//     //     if(abs(input_data[i]-decp_data[i])>bound){
//     //         std::cout<<abs(input_data[i]-decp_data[i])<<std::endl;
//     //     }
//     // }
//     // exit(0);
//     // std::cout<<"djal"<<std::endl;
//     // 10412175,6513917
//     // 10117038,5735947
//     float calculatermappath = 0.0;
//     std::cout<<"map开始"<<std::endl;
//     mappath1(dev_m, dev_a, dev_b,finddirection, mappath_path, datatransfer, 1);
//     std::cout<<"map结束"<<std::endl;
    

//     mappath1(dev_q, dev_c, dev_d,finddirection, mappath_path, datatransfer);
//     std::cout<<"map结束"<<std::endl;
    
    
//     size_t size = or_label.size();
//     // std::ofstream outFile11("label/or_label_"+filename+".bin", std::ios::binary);

//     // if (!outFile11.is_open()) {
//     //     throw std::runtime_error("Unable to open file for writing.");
//     // }
//     // outFile11.write(reinterpret_cast<const char*>(&size), sizeof(size));

//     // if (size > 0) {
//     //     outFile11.write(reinterpret_cast<const char*>(or_label.data()), size * sizeof(int));
//     // }

//     // outFile11.close();
    
//     // std::ofstream outFile12("label/dec_label_"+filename+'_'+compressor_id+'_'+std::to_string(bound)+".bin", std::ios::binary);
//     // // // 写入大小
//     // outFile12.write(reinterpret_cast<const char*>(&size), sizeof(size));

//     // // 写入vector的数据
//     // std::cout<<size<<std::endl;
//     // if (size > 0) {
//     //     outFile12.write(reinterpret_cast<const char*>(dec_label.data()), size * sizeof(int));
//     // }
//     // std::cout<<"写完了"<<std::endl;
    
//     // // exit(0);
//     // // 关闭文件
//     // outFile12.close();
    
//     auto startt = std::chrono::high_resolution_clock::now();
//     double right_labeled_ratio = 1-get_wrong_index_path();
//     // std::cout<<right_labeled_ratio<<std::endl;
//     // exit(0);
//     auto end = std::chrono::high_resolution_clock::now();
//     duration = end - startt;
    
//     getfpath+=duration.count();
//     // std::vector<double>* dev_f = &decp_data;
//     std::vector<double> decp_data_c1(decp_data); 

//     std::vector<double>* dev_k = &decp_data_c1;
//     // float counter_getfcp = 0.0;
//     float counter_fixtime_cp = 0.0;
//     // float counter_finddirection = 0.0;
//     // for(int i=0;i<10;i++){
//     //     std::cout<<i<<std::endl;
//     //     decp_data_c1 = decp_data;
//     double data_transfer_temp = datatransfer;
//     double fixtime_cp_temp = fixtime_cp;
//     double finddirection_temp = finddirection;
//     double getfcp_temp = getfcp;
//     double mappath_path_temp = mappath_path;
//     auto start3 = std::chrono::high_resolution_clock::now();
//     double getfpath_sub = 0;
//     double whole = 0;
    
//     double fixtime_path_sub = 0;
//     int cpite = 0;
    
//     std::vector<double> temp_time1;
//     startt = std::chrono::high_resolution_clock::now();
//     fix_process(dev_c,dev_d,dev_f,datatransfer, finddirection, getfcp, fixtime_cp,cpite);
//     duration = std::chrono::high_resolution_clock::now()-startt;
//     std::cout<<"fix: "<<fixtime_cp<<std::endl;
    
//     ite +=1;
//     // }
//     // exit(0);
//     //     std::cout<<"1000find:"<<counter_finddirection<<std::endl;
//     //     std::cout<<"1000getfcp:"<<counter_getfcp<<std::endl;
//     //     std::cout<<"1000fixcp:"<<counter_fixtime_cp<<std::endl;
//     // }
//     // std::cout<<"1000find:"<<counter_finddirection<<std::endl;
//     // std::cout<<"1000getfcp:"<<counter_getfcp<<std::endl;
//     // std::cout<<"10fixcp:"<<counter_fixtime_cp<<std::endl;
//     // exit(0);
//     // std::cout<<"djaldjas"<<std::endl;
    
    
    
//     // init_or_data(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f,size2);
        
//     // std::cout<<"第一次修复结束"<<std::endl;
//     // // exit(0);
//     // end1 = std::chrono::high_resolution_clock::now();
//     // end2 = std::chrono::high_resolution_clock::now();
//     // duration1 = end1 - start1;
//     // // fixtime += duration1.count();
//     // duration1 = end2 - start2;
//     // fixtime_cp += duration1.count();
    
//     // start2 = std::chrono::high_resolution_clock::now();
        
        
//     // };
//     // de_mesh.get_criticle_points();
    
        
    
//     // #pragma omp parallel for
//     // for (int i=0;i<size2;++i){
//     //     if(lowGradientIndices[i] == 1){
//     //         continue;
//     //     }
//     //         de_direction_as[i] = find_direction(i,decp_data,0);
        
//     //         de_direction_ds[i] = find_direction(i,decp_data,1);
        
//     // };
//     // for(int index=0;index<size2;index++){
//     //     if(de_direction_ds[index]!=-1){
//     //         dec_label[index*2] = index;
//     //     }
//     //     else{
//     //         dec_label[index*2] = -1;
//     //     }

//     //     if(de_direction_as[index]!=-1){
//     //         dec_label[index*2+1] = index;
//     //     }
//     //     else{
//     //         dec_label[index*2+1] = -1;
//     //     }
//     // }
    
//     // while(un_sign_as>0 or un_sign_ds>0){
//     //     un_sign_as=0;
//     //     un_sign_ds=0;
//     //     for(int i=0;i<size2;i++){
//     //         // std::cout<<i<<std::endl;
//     //         getlabel(i);
//     //     }
//     //     // std::cout<<un_sign_ds<<","<<un_sign_as<<std::endl;
        
//     // }
//     // for(int i=1;i<size2*2;i+=2){
//     //     int label1 = dec_label[i];
//     //     if(de_direction_as[label1]!=-1 and label1!=-1){
//     //         std::cout<<i<<", "<<label1<<", "<<de_direction_as[label1]<<std::endl;
//     //     }
//     // }

    
    
    
//     // for(int i=0;i<size2;i++){
//     //     if(de_direction_as[i]==-1){
//     //         std::cout<<i<<std::endl;
//     //     }
//     // }
//     // exit(0);
//     dev_a = &or_direction_as;
//     dev_b = &or_direction_ds;
//     dev_c = &de_direction_as;
//     dev_d = &de_direction_ds;
//     // for(int i=0;i<1000;i++){
//     mappath1(dev_q, dev_c, dev_d, finddirection, mappath_path, datatransfer);
//     // }
    
//     // exit(0);
//     // std::cout<<"djaldjas"<<std::endl;
    
//     // dec_label = mappath(de_mesh,de_direction_as,de_direction_ds);
//     startt = std::chrono::high_resolution_clock::now();
//     get_wrong_index_path();
//     // get_false_criticle_points();
    
    
    
//     end = std::chrono::high_resolution_clock::now();
//     duration = end - startt;
//     getfpath+=duration.count();
//     getfpath_sub+=duration.count();
//     temp_time1.push_back(getfcp-getfcp_temp);
//     temp_time1.push_back(fixtime_cp-fixtime_cp_temp);
//     // temp_time_ratio.push_back(fixtime_cp/whole);
//     temp_time1.push_back(fixtime_path_sub);
//     // temp_time_ratio.push_back(fixtime_path/whole);
//     temp_time1.push_back(finddirection - finddirection_temp);
//     // temp_time_ratio.push_back(searchdirection_time/whole);
//     temp_time1.push_back(mappath_path-mappath_path_temp);
//     // temp_time_ratio.push_back(searchtime/whole);
//     temp_time1.push_back(getfpath_sub);
//     duration = end - start3;
//     temp_time1.push_back(duration.count());
//     temp_time1.push_back(cpite);
//     time_counter.push_back(temp_time1);
//     while (wrong_index_as.size()>0 or wrong_index_ds.size()>0 or count_f_max>0 or count_f_min>0){
//         std::vector<double> temp_time;

//         data_transfer_temp = datatransfer;
//         fixtime_cp_temp = fixtime_cp;
//         finddirection_temp = finddirection;
//         getfcp_temp = getfcp;
//         mappath_path_temp = mappath_path;
        
//         getfpath_sub = 0;
//         whole = 0;
        
//         fixtime_path_sub = 0;
//         // searchdirection_time = 0;
//         // searchtime = 0;
//         // getfcp = 0;
//         // get_path = 0;
//         // whole = 0;
//         // std::vector<float> temp_time;
//         // std::vector<float> temp_time_ratio;

//         std::cout<<"修复路径: "<<wrong_index_as.size()<<","<<wrong_index_ds.size()<<std::endl;
        
        
//         // startt = std::chrono::high_resolution_clock::now();
        
        
//         // if(wrong_index_as.size()==1050){
//         //     for(int i:wrong_index_as){
//         //         std::cout<<i<<std::endl;
//         //     }
//         // }
//         startt = std::chrono::high_resolution_clock::now();
//         auto start_sub = std::chrono::high_resolution_clock::now();
//         // #pragma omp parallel for
//         for(int i =0;i< wrong_index_as.size();i++){
//             int j = wrong_index_as[i];
//             if(lowGradientIndices[j] == 1)
//             {
//                 continue;
//                 }
//             // std::cout<<j<<std::endl;
//             fixpath(j,0);
//         };
//         // std::cout<<"danfjka1"<<std::endl;
//         // #pragma omp parallel for
//         for(int i =0;i< wrong_index_ds.size();i++){
//             int j = wrong_index_ds[i];
//             if(lowGradientIndices[j] == 1){
//                 continue;}
//             // std::cout<<j<<std::endl;
//             fixpath(j,1);
//         };
        
        
//         end = std::chrono::high_resolution_clock::now();
//         duration = end - startt;
//         fixtime_path+=duration.count();
//         fixtime_path_sub+=duration.count();
//         // std::cout<<"danfjka"<<std::endl;
        
        
        
//         // get_false_criticle_points();
        
        
//         // searchdirection_time+=duration1.count();
        
   
        
//         //start2 = std::chrono::high_resolution_clock::now();
        
//         int cpite = 0;
        
//         // while(count_f_max>0 or count_f_min>0){
//         //     // std::cout<<"修复路径时:"<<count_f_max<<", "<<count_f_min<<std::endl;
            
//         //     cpite +=1;
//         //     start5 = std::chrono::high_resolution_clock::now();
//         //     start4 = std::chrono::high_resolution_clock::now();
//         // //     if(count_f_max ==0 and count_f_min==5){
//         // //     for(int i=0;i<count_f_min;i++){
//         // //         std::cout<<all_min[i]<<"," ;
//         // //     }
//         // //     std::cout<<std::endl;
//         // // }
//         //     #pragma omp parallel for
//         //     for(auto i = 0; i < count_f_max; i ++){
//         //         int critical_i = all_max[i];
//         //         fix_maxi_critical(critical_i,0);
//         //     }
            
            
            
//         //     #pragma omp parallel for
//         //     for(auto i = 0; i < count_f_min; i ++){
//         //         int critical_i = all_min[i];
//         //         fix_maxi_critical(critical_i,1);
//         //     }
            
            
            
//         //     end4 = std::chrono::high_resolution_clock::now();
//         //     d = end4-start4;
//         //     fixtime_cp += d.count();
            
            
        
//         //     #pragma omp parallel for
//         //     for (int i=0;i<size2;++i){
//         //         if(lowGradientIndices[i] == 1){
//         //             continue;
//         //         }
//         //             de_direction_as[i] = find_direction(i,decp_data,0);
                
//         //             de_direction_ds[i] = find_direction(i,decp_data,1);
                
//         //     };
//         //     start2 = std::chrono::high_resolution_clock::now();
            
//         //     get_false_criticle_points();
//         //     end2 = std::chrono::high_resolution_clock::now();
//         //     duration1 = end2 - start2;
//         //     getfcp += duration1.count();
        

//         // }
        
//         // dev_c = &de_direction_as;
//         // dev_d = &de_direction_ds;
//         // if(wrong_index_ds.size()==4){
//         //     std::cout<<"之前："<<std::endl;
//         //     std::cout<<de_direction_ds[8058]<<", "<<or_direction_ds[8058]<<std::endl;
//         // }
//         dev_f = &decp_data;
        
//         fix_process(dev_c,dev_d,dev_f,datatransfer, finddirection, getfcp, fixtime_cp,cpite);
        
        
//         // start2 = std::chrono::high_resolution_clock::now();
//         // de_mesh.values = decp_data;
//         // #pragma omp parallel for
//         // for (int i=0;i<size2;++i){
//         //     if(lowGradientIndices[i] == 1){
//         //         continue;
//         //     }
//         //     de_direction_as[i] = find_direction(i,decp_data,0);
//         //     de_direction_ds[i] = find_direction(i,decp_data,1);
//         // };
        
        
//         // dec_label = mappath(de_mesh,de_direction_as,de_direction_ds);
        
//         // end2 = std::chrono::high_resolution_clock::now();
//         // duration1 = end2 - start2;
        
//         // searchdirection_time+=duration1.count();
        
        
        
//         // std::cout<<de_direction_ds[3030]<<std::endl;
        
//         dev_q = &dec_label;
//         mappath1(dev_q,dev_c, dev_d,finddirection, mappath_path, datatransfer);
        
//         startt = std::chrono::high_resolution_clock::now();
        
//         get_wrong_index_path();
//         end = std::chrono::high_resolution_clock::now();
//         duration = end - startt;
//         getfpath+=duration.count();
//         getfpath_sub+=duration.count();
//         // end2 = std::chrono::high_resolution_clock::now();
//         // duration1 = end2-start2;
//         // get_path+=duration1.count();
//         // duration1 = end2-startt;
//         // whole += duration1.count();
//         // duration1 = end1-start1;
        
//         // std::cout<<"dfahlhql: "<<whole<<std::endl;
//         // temp_time.push_back(getfcp);
//         // temp_time_ratio.push_back(getfcp/whole);
//         // temp_time.push_back(fixtime_cp);
//         // temp_time_ratio.push_back(fixtime_cp/whole);
//         // temp_time.push_back(fixtime_path);
//         // temp_time_ratio.push_back(fixtime_path/whole);
//         // temp_time.push_back(searchdirection_time);
//         // temp_time_ratio.push_back(searchdirection_time/whole);
//         // temp_time.push_back(searchtime);
//         // temp_time_ratio.push_back(searchtime/whole);
//         // temp_time.push_back(get_path);
//         // temp_time_ratio.push_back(get_path/whole);
//         // record1.push_back(temp_time);
//         // record_ratio.push_back(temp_time_ratio);
//         auto end_sub = std::chrono::high_resolution_clock::now();
//         duration = end_sub-start_sub;
//         temp_time.push_back(getfcp-getfcp_temp);
//         temp_time.push_back(fixtime_cp-fixtime_cp_temp);
//         // temp_time_ratio.push_back(fixtime_cp/whole);
//         temp_time.push_back(fixtime_path_sub);
//         // temp_time_ratio.push_back(fixtime_path/whole);
//         temp_time.push_back(finddirection - finddirection_temp);
//         // temp_time_ratio.push_back(searchdirection_time/whole);
//         temp_time.push_back(mappath_path-mappath_path_temp);
//         // temp_time_ratio.push_back(searchtime/whole);
//         temp_time.push_back(getfpath_sub);

//         temp_time.push_back(duration.count());
//         temp_time.push_back(cpite);
//         time_counter.push_back(temp_time);
//         ite+=1;
//     };
//     // std::cout<<"修复结束"<<std::endl;
//     end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration1 = end - start;
//     // std::ofstream outFilep("result/performance1_cuda_"+filename+"_"+std::to_string(bound)+"_"+compressor_id+".txt", std::ios::app);
//     // // 检查文件是否成功打开
//     // if (!outFilep) {
//     //     std::cerr << "Unable to open file for writing." << std::endl;
//     //     return 1; // 返回错误码
//     // }
//     // // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
//     // // outFilep << std::to_string(number_of_thread)<<":" << std::endl;
//     // outFilep << "duration: "<<duration1.count() << std::endl;
//     // outFilep << "getfcp: "<<getfcp << std::endl;
    
//     // outFilep << "fixtime_cp: "<<fixtime_cp << std::endl;
    
//     // outFilep << "fixtime_path: " << fixtime_path << std::endl;
//     // outFilep << "mappath: " << mappath_path << std::endl;
    
//     // outFilep << "finddirection: " << finddirection << std::endl;
    
//     // outFilep << "getfpath:" << getfpath << std::endl;
//     // outFilep << "iteration number:" << ite+1 << std::endl;
//     // // outFilep << "edit_ratio: "<< ratio << std::endl;  
//     // int c = 0;  
//     // for (const auto& row : time_counter) {
//     //     outFilep << "iteration: "<<c<<": ";
//     //     for (size_t i = 0; i < row.size(); ++i) {
//     //         outFilep << row[i];
//     //         if (i != row.size() - 1) { // 不在行的末尾时添加逗号
//     //             outFilep << ", ";
//     //         }
//     //     }
//     //     // 每写完一行后换行
//     //     outFilep << std::endl;
//     //     c+=1;
//     // }
//     // outFilep << "\n"<< std::endl;
    
    
    
//     // // return 0;
//     // // de_mesh.get_criticle_points();
    
//     std::cout<<"duration: "<<duration1.count()<<std::endl;
//     // std::cout<<"data_transfer:"<<datatransfer<<std::endl;
//     // std::cout<<"find_direction:"<<finddirection<<std::endl;
//     // std::cout<<"getfcp:"<<getfcp<<std::endl;
//     // std::cout<<"mappath_path:"<<mappath_path<<std::endl;
//     // std::cout<<"getfpath:"<<getfpath*1000<<std::endl;
//     // std::cout<<"fixfcp:"<<fixtime_cp<<std::endl;
//     // std::cout<<"fixpath:"<<fixtime_path*1000<<std::endl;
//     std::ofstream outFile13("label/fixed_dec_label_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin");

//     if (!outFile13.is_open()) {
//         throw std::runtime_error("Unable to open file for writing.");
//     }

//     // 获取vector的大小
    

//     // 写入大小
//     outFile13.write(reinterpret_cast<const char*>(&size), sizeof(size));

//     // 写入vector的数据
//     if (size > 0) {
//         outFile13.write(reinterpret_cast<const char*>(&dec_label[0]), size * sizeof(int));
//     }

//     // 关闭文件
//     outFile13.close();
//     // return 0;
   
//     // std::cout << "fixtime: " << fixtime << " seconds" << std::endl;
//     // std::cout << "get_fcp: " << getfcp/duration.count() << " seconds" << std::endl;
//     // std::cout << "fixtime_cp: " << fixtime_cp/duration.count() << " seconds" << std::endl;
//     // std::cout << "fixtime_path: " << fixtime_path/duration.count() << " seconds" << std::endl;
//     // std::cout << "searchtime: " << searchtime << " seconds" << std::endl;
//     // std::cout << "finddirection: " << searchdirection_time/duration.count() << " seconds" << std::endl;
//     // std::cout << "getfpath:" << get_path/duration.count() << std::endl;
//     // std::cout << "iteration number:" << ite << std::endl;
//     // std::ofstream file2("dec_label_iteration内"+std::to_string(width)+".txt");
//     // result = "";
//     // for (const auto& p : dec_label) {
//     //     result+="[";
//     //     result+=std::to_string(p.first)+","+std::to_string(p.second);
//     //     result+="],";
//     // }
//     // file2 << result << "\n";
//     // file2 << "#"<<",";
    
//     // result = "[";
//     // for( int i:de_mesh.maxi){
        
//     //     result+=std::to_string(i);
//     //     result+=",";
//     // }
//     // result+="]";
//     // file2 << result;
//     // file2 << "\n";
//     // file2 << "#"<<",";
//     // result = "[";
//     // for( int i:de_mesh.mini){
        
//     //     result+=std::to_string(i);
//     //     result+=",";
//     // }
//     // result+="]";
//     // file2 << result;

//     // // // 关闭文件
//     // file2.close();
//             // std::ofstream file1("or_label.txt");

//     // // 写入数据
    
//     // for (const auto& p : or_label) {
        
//     //     file1 << p.first << ", " << p.second << "\n";
//     // }
    
    
//     // file1 << "#"<<",";
//     // for( int i:or_mesh.maxi){
//     //     file1 << i << ",";
//     // }
//     // file1 << "\n";
//     // file1 << "#"<<",";
//     // for( int i:or_mesh.mini){
//     //     file1 << i << ",";
//     // }
//     // // 关闭文件
//     // file1.close();
//     // std::ofstream outFile("minlabel_"+std::to_string(width)+".txt");

//     // if (outFile.is_open()) {
//     //     outFile << std::setprecision(std::numeric_limits<double>::max_digits10);

//     //     for (int i=0;i<decp_data.size();i++) {
            
//     //         outFile << dec_label[i*2] << std::endl;
//     //     }
//     //     outFile.close();
//     // } else {
//     //     std::cerr << "Unable to open the file for writing." << std::endl;
//     // }

//     // std::ofstream outFile1("original_minlabel_"+std::to_string(width)+".txt");

//     // if (outFile1.is_open()) {
//     //     outFile1 << std::setprecision(std::numeric_limits<double>::max_digits10);

//     //     for (int i=0;i<decp_data.size();i++) {
            
//     //         outFile1 << or_label[i*2] << std::endl;
//     //     }
//     //     outFile1.close();
//     // } else {
//     //     std::cerr << "Unable to open the file for writing." << std::endl;
//     // }
//     std::ofstream outFile(fix_path);
//     // std::ofstream outFile("values.bin", std::ios::binary | std::ios::out);
//     if (outFile.is_open()) {
//         outFile.write(reinterpret_cast<const char*>(decp_data.data()), decp_data.size() * sizeof(double));
//     }
//     outFile.close();
//     // start = std::chrono::high_resolution_clock::now();
//     std::vector<double> indexs;
//     std::vector<double> edits;
//     for (int i=0;i<input_data.size();i++){
        
//         if (decp_data_copy[i]!=decp_data[i]){
//             indexs.push_back(i);
//             edits.push_back(decp_data[i]-decp_data_copy[i]);
//             cnt++;
//         }
//     }
//     std::vector<int> diffs;  // 存储差值的 vector
//     std::string indexfilename = "data"+filename+".bin";
//     std::string editsfilename = "data_edits"+filename+".bin";
//     std::string compressedindex = "data"+filename+".bin.zst";
//     std::string compressededits = "data_edits"+filename+".bin.zst";
//     // 在计算差值前将第一个元素添加到 diffs 中
//     if (!indexs.empty()) {
//         diffs.push_back(indexs[0]);
//     }
//     for (size_t i = 1; i < indexs.size(); ++i) {
//         diffs.push_back(indexs[i] - indexs[i - 1]);
//     }
//     double ratio = double(cnt)/(decp_data_copy.size());
//     std::cout<<cnt<<","<<ratio<<std::endl;
//     std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
//     if (file.is_open()) {
//         file.write(reinterpret_cast<const char*>(diffs.data()), diffs.size() * sizeof(int));
//     }
//     file.close();
//     command = "zstd -f " + indexfilename;
//     std::cout << "Executing command: " << command << std::endl;
//     result = std::system(command.c_str());
//     if (result == 0) {
        
//         std::cout << "Compression successful." << std::endl;
//     } else {
//         std::cout << "Compression failed." << std::endl;
//     }
    
//     std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
//     if (file1.is_open()) {
//         file1.write(reinterpret_cast<const char*>(edits.data()), edits.size() * sizeof(double));
//     }
//     file1.close();
//     command = "zstd -f " + editsfilename;
//     std::cout << "Executing command: " << command << std::endl;
//     result = std::system(command.c_str());
//     if (result == 0) {
//         std::cout << "Compression successful." << std::endl;
//     } else {
//         std::cout << "Compression failed." << std::endl;
//     }
//     // compute result
//     double compressed_indexSize = fs::file_size(compressedindex);
//     double compressed_editSize = fs::file_size(compressededits);
//     double original_indexSize = fs::file_size(indexfilename);
//     double original_editSize = fs::file_size(editsfilename);
//     double original_dataSize = fs::file_size(inputfilename);
    
//     // std::cout<<compressed_indexSize<<", "<<original_indexSize<<", "<<original_indexSize/compressed_indexSize<<std::endl;
//     // std::cout<<compressed_editSize<<", "<<original_editSize<<", "<<original_editSize/compressed_editSize<<std::endl;
//     // std::cout<<compressed_dataSize<<", "<<original_dataSize<<", "<<original_dataSize/compressed_dataSize<<std::endl;
//     double overall_ratio = (original_indexSize+original_editSize+original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
//     double bitRate = 64/overall_ratio; 
    
    


//     double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
//     double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);
//     std::cout<<psnr<<", "<<fixed_psnr<<std::endl;
//     std::cout<<"right: "<<right_labeled_ratio<<std::endl;
//     std::cout<<"relative range: "<<range<<std::endl;
    
//     std::ofstream outFile3("../result/result_"+filename+"_"+compressor_id+"tmp3_detailed.txt", std::ios::app);

//     // 检查文件是否成功打开
//     if (!outFile3) {
//         std::cerr << "Unable to open file for writing." << std::endl;
//         return 1; // 返回错误码
//     }

    
//     outFile3 << std::to_string(bound)<<":" << std::endl;
//     outFile3 << std::setprecision(17)<< "related_error: "<<range << std::endl;
//     outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
//     outFile3 << std::setprecision(17)<<"CR: "<<original_dataSize/compressed_dataSize << std::endl;
//     outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
//     outFile3 << std::setprecision(17)<<"BR: "<< (compressed_dataSize*8)/size2 << std::endl;
//     outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
//     outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;

//     outFile3 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
//     outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
//     outFile3 << std::setprecision(17)<<"relative: "<<range << std::endl;
//     outFile3 << "\n" << std::endl;
//     // 关闭文件
//     outFile3.close();

//     std::cout << "Variables have been appended to output.txt" << std::endl;

//     // std::cout<<overall_ratio * bitRate<<std::endl;
//     // std::cout<<overall_ratio<<","<<bitRate<<std::endl;

    

    
//     // std::vector<double> diff(input_data.size());

//     // // 计算差值的绝对值
//     // std::transform(input_data.begin(), input_data.end(), decp_data.begin(), diff.begin(),
//     //                [](double a, double b) {
//     //                    return std::abs(a - b);
//     //                });

//     // 找出最大的绝对差值
//     // auto max_diff = *std::max_element(diff.begin(), diff.end());

//     // std::cout << "The largest absolute difference is: " << max_diff << std::endl;
//     // std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
//     // std::cout<<cnt<<","<<ratio<<std::endl;
    
//     // std::cout << "get_fcp: " << getfcp << " seconds" << std::endl;
//     // std::cout << "fixtime_cp: " << fixtime_cp << " seconds" << std::endl;
//     // std::cout << "fixtime_path: " << fixtime_path << " seconds" << std::endl;
//     // std::cout << "searchtime: " << searchtime << " seconds" << std::endl;
//     // std::cout << "finddirection: " << searchdirection_time << " seconds" << std::endl;
//     // std::cout << "getfpath:" << get_path << std::endl;
//     // std::cout << "iteration number:" << ite << std::endl;
    
//     // std::string result = "[";
//     // for(int i=0;i<record1.size();i++){
//     //     std::string result1 = "[";
//     //     std::vector<float> temp = record1[i];
//     //     for(auto k:temp){
//     //         result1+=std::to_string(k);
//     //         result1+=",";
//     //     }
//     //     result1+="]";
//     //     result1+=",";
//     //     result+=result1;
//     // }
//     // result+="]";
//     // std::cout<<result<<std::endl;
//     // result = "[";
//     // for(int i=0;i<record_ratio.size();i++){
//     //     std::string result1 = "[";
//     //     std::vector<float> temp = record_ratio[i];
//     //     for(auto k:temp){
//     //         result1+=std::to_string(k);
//     //         result1+=",";
//     //     }
//     //     result1+="]";
//     //     result1+=",";
//     //     result+=result1;
//     // }
//     // result+="]";
//     // std::cout<<result<<std::endl;
    
//     // std::system("rm -f "+decpfilename);
//     if(mode==1){
//         if(bitRate==target_br){
//             std::string path1 ="decompressed_data/fixed_decp_"+filename+"_"+std::to_string(bound)+"target_"+std::to_string(bitRate)+".bin";
//             std::ofstream outFile4(path1);
//             // std::ofstream outFile("values.bin", std::ios::binary | std::ios::out);
//             if (outFile4.is_open()) {
//                 outFile4.write(reinterpret_cast<const char*>(decp_data.data()), decp_data.size() * sizeof(double));
//             }
//             outFile4.close();
//             std::ofstream outFile5("../result/result_"+filename+"_"+compressor_id+".txt", std::ios::app);

//             // 检查文件是否成功打开
//             if (!outFile5) {
//                 std::cerr << "Unable to open file for writing." << std::endl;
//                 return 1; // 返回错误码
//             }

            
//             outFile5 << filename <<"_"<<std::to_string(bound)<<":" << std::endl;
//             outFile5 << std::setprecision(17)<<"relative_error: "<<range << std::endl;
//             outFile5 << std::setprecision(17)<<"OCR: "<<overall_ratio << std::endl;
//             outFile5 << std::setprecision(17)<<"CR: "<<original_dataSize/compressed_dataSize << std::endl;
//             outFile5 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
//             outFile5 << std::setprecision(17)<<"BR: "<< (compressed_dataSize*8)/size2 << std::endl;
//             outFile5 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
//             outFile5 <<std::setprecision(17)<< "fixed_psnr: "<<fixed_psnr << std::endl;

//             outFile5 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
//             outFile5 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
//             outFile5 << "\n" << std::endl;
//             // 关闭文件
//             outFile5.close();
//             std::cout << "target_bitrate_founded" << std::endl;
//         }
//     }
//     return 0;
// }

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <stdatomic.h>
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
#include <cstdio>
// #include <vtkSmartPointer.h>
// #include <vtkImageData.h>
// #include <vtkIntArray.h>
// #include <vtkPointData.h>
// #include <vtkXMLImageDataReader.h>
// #include <vtkXMLImageDataWriter.h>
using namespace std;
namespace fs = std::filesystem;
int pre = 0;
// g++ -std=c++17 -O3 -g hello.cpp -o helloworld
// g++ -std=c++17 -O3 -g -fopenmp -c preserve3d.cpp -o hello2.o
// g++-12 -fopenmp hello2.o kernel.o -lcudart -o helloworld
// g++-12 -fopenmp -std=c++17 -O3 -g hello2.cpp -o helloworld2
// 4.5177893520000003
// g++ -fopenmp hello2.o kernel.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld

int width;
int height;
int depth;
int size2;
int un_sign_as;
int un_sign_ds;
int ite = 0;
std::vector<int> all_max, all_min, all_d_max, all_d_min;
atomic_int count_max = 0;
atomic_int count_min = 0;
atomic_int count_f_max = 0;
atomic_int count_f_min = 0;

std::vector<int> record;
std::vector<std::vector<float>> record1;
std::vector<std::vector<float>> record_ratio;
// 没有0，0 和 1，1
std::vector<std::vector<int>> directions1 = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};

// std::vector<double> getdata(std::string filename){
    
//      std::vector<double> data;
//      data.resize(width*height*depth);
//      std::string line;
//      std::ifstream file(filename);
//     //  std::ifstream file(filename);
//     // cout<<"dh"<<endl;
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
//     // cout<<"读完了"<<cnt<<endl;
//     file.close();
    
//     // cout<<data.size()<<endl;
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
//     //                 cout<<value<<endl;
//     //             }
                
//     //             std::cerr << "转换错误: " << e.what() << '\n';
//     //             // 异常处理代码
//     //         }

//     //         // cout<<value<<endl;
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
    // cout<<num_floats<<endl;
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
    // return lowGradientIndices;
    const double threshold = 1e-16; // 梯度阈值
    // 遍历三维数据计算梯度
    for (int i = 0; i < width; ++i) {
        
        for (int j = 0; j < height; ++j) {
            
            for (int k = 0; k < depth; ++k) {
                
                int rm = i  + j * width + k * (height * width);
                
                
                // for(int q=0;q<12;q++){
                for (auto& dir : directions1) {
                    int newX = i + dir[0];
                    int newY = j + dir[1];
                    int newZ = k + dir[2];
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
    // cout<<"ok"<<endl;
    return lowGradientIndices;
}

std::vector<int> lowGradientIndices;

std::vector<std::vector<int>> _compute_adjacency(){
    std::vector<std::vector<int>> adjacency;
    for (int i = 0; i < size2; ++i) {
            int y = (i / (width)) % height; // Get the x coordinate
            int x = i % width; // Get the y coordinate
            int z = (i / (width * height)) % depth;
            std::vector<int> adjacency_temp;
            for (auto& dir : directions1) {
                int newX = x + dir[0];
                int newY = y + dir[1];
                int newZ = z + dir[2];
                int r = newX + newY  * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                
                
                // Check if the new coordinates are within the bounds of the mesh
                if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0 && lowGradientIndices[r] != 1) {
                    
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
std::vector<std::vector<int>> adjacency;

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
        void get_criticle_points();
        // std::vector<int> adjacent_vertices( int index);

}; 

void myTriangularMesh::get_criticle_points(){
    
    std::vector<int> global_maxi_temp;
    std::vector<int> global_mini_temp;

    #pragma omp parallel
    {
        std::vector<int> local_maxi_temp;
        std::vector<int> local_mini_temp;

        #pragma omp for nowait
        for (int i = 0; i < size2; ++i) {
            if(lowGradientIndices[i] == 1){
                continue;
            }
            bool is_maxima = true;
            bool is_minima = true;

            for (int j : adjacency[i]) {
                if(lowGradientIndices[j] == 1){
                    continue;
                }
                if (this->values[j] > this->values[i]) {
                    is_maxima = false;
                    break;
                }
                else if(this->values[j] == this->values[i] and j>i){
                    is_maxima = false;
                    break;
                }
            }
            for (int j : adjacency[i]) {
                if(lowGradientIndices[j] == 1){
                    continue;
                }
                if (this->values[j] < this->values[i]) {
                    is_minima = false;
                    break;
                }
                else if(this->values[j] == this->values[i] and j<i){
                    is_minima = false;
                    break;
                }
            }

            if (is_maxima and lowGradientIndices[i] == 0) {
                local_maxi_temp.push_back(i);
            }
            if (is_minima and lowGradientIndices[i] == 0) {
                local_mini_temp.push_back(i);
            }
        }

        #pragma omp critical
        {
            global_maxi_temp.insert(global_maxi_temp.end(), local_maxi_temp.begin(), local_maxi_temp.end());
            global_mini_temp.insert(global_mini_temp.end(), local_mini_temp.begin(), local_mini_temp.end());
        }
    }

    // std::unordered_set<int> mesh_maxi(global_maxi_temp.begin(), global_maxi_temp.end());
    // std::unordered_set<int> mesh_mini(global_mini_temp.begin(), global_mini_temp.end());
    // #pragma omp parallel for
    //     for (int i = 0; i < size2; ++i) {
    //         if (maximum[i] == -1) {
    //             // #pragma omp critical
    //             maxi_temp.push_back(i);
    //         }
    //     }
    // // #pragma omp parallel for
    //     for (int i = 0; i < size2; ++i) {
    //         if (minimum[i] == -1) {
    //             // #pragma omp critical
    //             mini_temp.push_back(i);
    //         }
    //     }
    std::unordered_set<int> mesh_maxi(global_maxi_temp.begin(), global_maxi_temp.end());
    std::unordered_set<int> mesh_mini(global_mini_temp.begin(), global_mini_temp.end());
    this->maxi = mesh_maxi;
    this->mini = mesh_mini;
    

};

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
int direction_to_index_mapping[12][3] = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};

int getDirection(const std::map<std::tuple<int, int, int>, int>& direction_mapping, int row_diff, int col_diff, int dep_diff) {
    auto it = direction_mapping.find(std::make_tuple(row_diff, col_diff,dep_diff));
    
    if (it != direction_mapping.end()) {
        return it->second;
    } else {
        return -1; 
    }
}
// int from_direction_to_index(int cur, int direc){
//     if (direc==-1) return cur;
//     int x = cur % width;
//     int y = (cur / width) % height;
//     int z = (cur/(width * height))%depth;
    
//     auto it = reverse_direction_mapping.find(direc);
//     if (it != reverse_direction_mapping.end()) {
//     // 找到了元素，it->first 是键，it->second 是值
//         std::tuple<int, int,int> value = it->second;
        
//         int delta_row = std::get<0>(value); 
//         int delta_col = std::get<1>(value); 
//         int delta_dep = std::get<2>(value); 
//         int next_row = x + delta_row;
//         int next_col = y + delta_col;
//         int next_dep = z + delta_dep;
        
//         return next_row + next_col  * width + next_dep* (height * width);
//     } else {
//         return -1;
//     }
    
// };
int from_direction_to_index(int cur, int direc){
    
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
//                 //     cout<<index<<", "<<largetst_index<<","<<decp_data[largetst_index]<<", "<<decp_data[113667]<<endl;
//                 // }

                
//             };
//         };
//     };
    
//     // if(lowGradientIndices[largest_index] == 1){
//     //     cout<<largetst_index<<endl;
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
// int find_direction1 (int index,std::map<int,int> &data ,int direction){

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
//     // cout<<largetst_index<<","<<index<<endl;
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
//             //     cout<<cur<<", "<<i<<endl;
//             // }
//             int direc = direction_as[cur];
            
//             int rank1 = (cur / (height)) % width;
//             int row1 = cur%height;
//             int depth1 = cur/(width * height);
            
//             int next_vertex = from_direction_to_index(cur, direc);
            
            
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
            
//             int next_vertex = from_direction_to_index(cur, direc);
            

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
std::vector<int> or_direction_as;
std::vector<int> or_direction_ds;
std::vector<int> de_direction_as;
std::vector<int> de_direction_ds;

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
            
//             int next_vertex = from_direction_to_index(index,or_direction_as[index]);
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
//                         //             if(smallest_vertex==66783){cout<<"在这里11."<<endl;}
//                         //             decp_data[smallest_vertex]-=diff2;
//                         //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
//                         //         }
                                
                                
//                         //     }
                            
//                         // }

//                         if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex] and d>=1e-16){
//                             // if(index==1620477){
//                             //     // cout<<"next_vertex: "<<decp_data[next_vertex]<<endl;
//                             //     // cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<endl;
//                             //     cout<<"before index: "<<decp_data[index]<<endl;
                                
//                             // }
                            
//                             decp_data[index] -= d;
                            
                            
                                            
//                         }
//                         // else if(abs(input_data[next_vertex]-(decp_data[next_vertex]+d1))<=bound and decp_data[index]>=decp_data[next_vertex] and d1>0){
//                         //     // if(index==1620477){
//                         //     //     // cout<<"next_vertex: "<<decp_data[next_vertex]<<endl;
//                         //     //     // cout<<"smallest_vertex: "<<decp_data[smallest_vertex]<<endl;
//                         //     //     cout<<"before index: "<<decp_data[index]<<endl;
                                
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
//                     //     cout<<"在这时候d: "<<d<<endl;
//                     // }   
//                     // double d = 1e-16;
//                     if(abs(input_data[index]-decp_data[index]+d)<=bound){
                        
//                         decp_data[index]-=d;
//                     }
//                     else if(abs(input_data[next_vertex]-decp_data[next_vertex]-d)<=bound){
//                         // if(next_vertex==78){cout<<"在这里21"<<endl;}
//                         decp_data[next_vertex]+=d;
//                     }
//                 }
                
//             }
            
            
        
//         }
//         else{
//             // if(index==25026 and count_f_max<=770){
//             //     cout<<"在这里"<<endl;
//             // }
            
//             int largest_index = from_direction_to_index(index,de_direction_as[index]);
//             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
//             // if(index==25026 and count_f_max<=770){
//             //     cout<<"改变前"<<endl;
//             //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
//             //     cout<<"next_vertex: "<<largest_index<<","<<decp_data[largest_index]<<endl;
//             //     // cout<<"smallest_vertex: "<<smallest_vertex<<", "<<decp_data[smallest_vertex]<<endl;
//             //     cout<<"diff: "<<d<<","<<"d: "<<d<<endl;
//             //     cout<<or_direction_as[25026]<<de_direction_as[25026]<<endl;
//             // }
//             if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
//                 de_direction_as[index] = -1;
//             }
//             if(d>=1e-16){
                
//                 if (decp_data[index]<=decp_data[largest_index]){
//                     if(abs(input_data[largest_index]-decp_data[index]+d)){
//                         // if(largest_index==66783){cout<<"在这里17"<<endl;}
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
//             int next_vertex= from_direction_to_index(index,or_direction_ds[index]);
            
//             double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
//             double d =  (bound+input_data[index]-decp_data[index])/2.0;
//             // double d1 =  (bound-(input_data[next_vertex]-decp_data[next_vertex]))/2.0;
            
//             double d1 = (decp_data[next_vertex]-input_data[next_vertex]+bound)/2.0;
//             if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
//                 de_direction_ds[index]=or_direction_ds[index];
//                 return 0;
//             }

//             // if(index == 6595 and count_f_min==5){
//             //     cout<<"下降："<<endl;
//             //     cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
//             //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
//             //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
//             //     cout<<"diff: "<<diff<<endl;
//             //     cout<<"d: "<<d<<endl;
//             //     cout<<"d1: "<<d1<<endl;
//             // }
            
//             if(diff>=1e-16 or d>=1e-16 or d1>=1e-16){
                
//                 if(decp_data[index]==decp_data[next_vertex]){
                    
                      
                    
//                         while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
//                             diff/=2;
//                         }
                        
//                         if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
//                             // if(index==344033 and count_f_min==2){cout<<"在这里22"<<d<<endl;}
//                             decp_data[next_vertex]= decp_data[index]-diff;
//                         }
//                         else if(d1>=1e-16){
//                             // if(index==344033 and count_f_min==2){cout<<"在这里23"<<d<<endl;}
//                             decp_data[next_vertex]-=d1;
//                         }
//                         else if(d>=1e-16){
//                             // if(index==344033 and count_f_min==2){cout<<"在这里24"<<d<<endl;}
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
//                                 // if(index==270808 and count_f_min==1){cout<<"在这里2！"<< endl;}
//                                 while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
//                                     diff*=2;
//                                 }
//                                 if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
//                                     decp_data[next_vertex] = decp_data[index]-diff;
//                                 }
//                                 // if(index == 6595 and count_f_min==5){
//                                 //     cout<<"在这里1！"<< diff <<", "<<index<<", "<<decp_data[index]<<","<<input_data[index]<<","<<input_data[next_vertex]<<endl;

//                                 // }
//                                 // if(next_vertex==66783){cout<<"在这里13"<<endl;}
//                                 // decp_data[next_vertex] = decp_data[index]-diff;
//                                 // if(index==89797){
//                                 //         cout<<"在这里2"<<diff<<", "<<d<<endl;
//                                 // }

//                                 // decp_data[index]+=d;
//                             }
//                             // else if(abs(input_data[index]-decp_data[index]-d)<=bound and decp_data[index]<=decp_data[next_vertex] and d>0){
//                             //     if(index==135569){cout<<"在这里23"<<endl;}
//                             //     decp_data[index]+=d;
//                             // }
//                             else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
//                                 while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-17){
//                                     d1*=2;
//                                 }
//                                 // if(count_f_min<=12){cout<<"在这里2！"<<abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<<"," <<d1<< endl;}
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
//                                 //if(index == 6595 and count_f_min==5){cout<<"在这里3！"<<abs(input_data[next_vertex]-bound-decp_data[next_vertex])<< endl;}
//                             }
                            
                            
                        
                        
//                 };

//                 }
                
                

                
//             }

//             else{
                
//                 if(decp_data[index]<decp_data[next_vertex]){
//                     // if(next_vertex==339928 and wrong_maxi_cp.size()==84){
//                     //     cout<<"np下降："<<endl;
//                     //     cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
//                     //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
//                     //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
//                     //     cout<<"diff: "<<diff<<endl;
//                     //     cout<<"d: "<<d<<endl;
                
//                     //     }
                        
//                         // if(abs(input_data[index]-(decp_data[next_vertex]))<=bound and abs(input_data[next_vertex]-decp_data[index])<=bound){
//                         //     double t = decp_data[index];
//                         //     decp_data[index] = decp_data[next_vertex];
//                         //     if(next_vertex==66783){cout<<"在这里14"<<endl;}
//                         //     decp_data[next_vertex] = t;
                            
//                         // }
//                         if(abs(input_data[next_vertex]-decp_data[index])<bound){
//                             double t = (decp_data[index]-(input_data[index]-bound))/2.0;
//                             // if(index==949999){cout<<"在这里24"<<endl;}
//                             // decp_data[index] = decp_data[next_vertex];
//                             // if(next_vertex==66783){cout<<"在这里14"<<endl;}
//                             decp_data[next_vertex] = decp_data[index]-t;
                            
//                         }
//                         else{
//                             //if(index==949999){cout<<"在这里29"<<endl;}
//                             decp_data[index] = input_data[index] + bound;
//                         }
//                 }
                
//                 else if(decp_data[index]==decp_data[next_vertex]){
//                     double d = (bound - (input_data[index]-decp_data[index]))/64;
//                     // while(abs(input_data[index]-decp_data[index]-d)>bound and d>=2e-16){
//                     //         d/=2;
//                     // }
//                     // if(index==949999){
//                     //     cout<<"在这里99 "<<d<<endl;
//                     // }   
//                     // double d = 1e-16;
//                     if(abs(input_data[index]-decp_data[index]-d)<=bound){
//                         decp_data[index]+=d;
//                     }
//                     else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
//                         // if(next_vertex==66783){cout<<"在这里13"<<endl;}
//                         decp_data[next_vertex]-=d;
//                     }
//                 }
//             }
            

            
            
            
//         // if(index == 6595 and count_f_min==5){
//         //         cout<<"下降后："<<endl;
//         //         cout<<"next: "<<next_vertex<<", "<<decp_data[next_vertex]<<endl;
//         //         cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
//         //         cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[next_vertex]<<endl;
//         //         cout<<"diff: "<<diff<<endl;
//         //         cout<<"d: "<<d<<endl;
//         //         cout<<"d1: "<<d1<<endl;
//         //         cout<<input_data[index]<<","<<input_data[next_vertex]<<endl;
//         //     }
            
        
//         }
    
//         else{
            
//             int largest_index = from_direction_to_index(index,de_direction_ds[index]);
//             double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
//             double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
//             // if(count_f_min==84){
//             //     cout<<"np下降："<<endl;
//             //     cout<<"next: "<<largest_index<<", "<<decp_data[largest_index]<<endl;
//             //     cout<<"index: "<<index<<", "<<decp_data[index]<<endl;
//             //     cout<<"daxiaoguanxi: "<<decp_data[index]-decp_data[largest_index]<<endl;
//             //     cout<<"diff: "<<diff<<endl;
//             //     cout<<"d: "<<d<<endl;
                
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
//                         //     cout<<"在这里2！"<<endl;
//                         // }
                        
//                         decp_data[index] -= diff;
//                     }
                    
                    
//                 }                    
//             }
            
                    
//             else{
//                 if (decp_data[index]>=decp_data[largest_index]){
                    
//                     // if(index==66783){cout<<"在这里15"<<endl;}
//                     decp_data[index] = input_data[index] - bound;
//                 }   
    
//             }


               
//         }

        
//     }    
//     return 0;
// };
// int fix_maxi_critical(int index, int direction){
    
//     if (direction == 0){
        
//         if (or_direction_as[index]!=-1){
            
//             int next_vertex = from_direction_to_index(index,or_direction_as[index]);
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
//             //         cout<<index<<","<<next_vertex<<","<<smallest_vertex<<endl;
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
//                                     // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
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
            
//             int largest_index = from_direction_to_index(index,de_direction_as[index]);
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
//             int next_vertex= from_direction_to_index(index,or_direction_ds[index]);
            
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
            
//             int largest_index = from_direction_to_index(index,de_direction_ds[index]);
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
//             int next_vertex =  from_direction_to_index(cur,de_direction_as[cur]);
            
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
            
//             int false_index= from_direction_to_index(cur,de_direction_as[cur]);
//             int true_index= from_direction_to_index(cur, or_direction_as[cur]);
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
//                                 // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
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
//                     //             // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
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
            
//             int next_vertex = from_direction_to_index(cur,de_direction_ds[cur]);
            
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
            
//             int false_index= from_direction_to_index(cur,de_direction_ds[cur]);
//             int true_index= from_direction_to_index(cur, or_direction_ds[cur]);
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
int fixpath(int index, int direction){
    if(direction==0){
        int cur = index;
        while (or_direction_as[cur] == de_direction_as[cur]){
            int next_vertex =  from_direction_to_index(cur,de_direction_as[cur]);
            
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
            
            int false_index= from_direction_to_index(cur,de_direction_as[cur]);
            int true_index= from_direction_to_index(cur, or_direction_as[cur]);
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
            
            for(int i:adjacency[false_index]){
                
                if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];

            double threshold1 = std::numeric_limits<double>::max();
            int smallest_vertex1 = true_index;
            
            for(int i:adjacency[true_index]){
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
                            
                            if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                                decp_data[smallest_vertex]-=diff2;
                                // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                            }
                            
                            
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
                        // ascending version;
                        decp_data[true_index] = decp_data[false_index] + diff;
                        // decp_data[false_index] = decp_data[true_index]-diff;
                    }
                    if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
                        // decp_data[true_index] +=d;
                        decp_data[false_index] -=d;
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
                        decp_data[false_index] = (decp_data[false_index]+input_data[true_index]-bound)/2.0;
                    }
                        
                    else{
                        decp_data[false_index] = input_data[false_index] - bound;
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
            
            int next_vertex = from_direction_to_index(cur,de_direction_ds[cur]);
            
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
            
            int false_index= from_direction_to_index(cur,de_direction_ds[cur]);
            int true_index= from_direction_to_index(cur, or_direction_ds[cur]);
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
                            decp_data[true_index] = decp_data[false_index] - diff;
                        }
                        if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                            decp_data[false_index] += d;
                        }
                        
                        
                        

                        // diff = (bound-abs(input_data[false_index]-decp_data[true_index]))/2.0;
                        
                        
                        
                        // diff = (bound-abs(input_data[true_index]-decp_data[false_index]))/2.0;
                        if (decp_data[false_index]==decp_data[true_index]){
                            if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                        
                                decp_data[false_index] += d;
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
                        decp_data[false_index] =  (decp_data[true_index]+input_data[true_index]+bound)/2.0;
                    }
                    else{
                        decp_data[false_index] = input_data[false_index] + bound;
                    }
                    while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound and diff>=2e-17){
                        diff/=2;
                    }
                    if (decp_data[false_index]==decp_data[true_index]){
                        double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                        decp_data[false_index]+=diff;
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
    // cout<<cnt/size2<<endl;
    double result = static_cast<double>(cnt) / static_cast<double>(size2);
    return result;
};
// bool compareIndices(int i1, int i2, const std::vector<double>& data) {
//     if (data[i1] == data[i2]) {
//         return i1<i2; // 注意这里是‘>’
//     }
    
//     return data[i1] < data[i2];
// }

// void get_false_criticle_points(){
//     count_max=0;
//     count_min=0;

//     count_f_max=0;
//     count_f_min=0;

//     #pragma omp parallel for
//     for (auto i = 0; i < size2; i ++) {
//             if(lowGradientIndices[i] == 1){
//                 continue;
//             }
//             bool is_maxima = true;
//             bool is_minima = true;

//             for (int j : adjacency[i]) {
//                 if(lowGradientIndices[j] == 1){
//                     continue;
//                 }
//                 if (decp_data[j] > decp_data[i]) {
//                     is_maxima = false;
//                     break;
//                 }
//                 else if(decp_data[j] == decp_data[i] and j>i){
//                     is_maxima = false;
//                     break;
//                 }
//             }
//             for (int j : adjacency[i]) {
//                 if(lowGradientIndices[j] == 1){
//                     continue;
//                 }
//                 if (decp_data[j] < decp_data[i]) {
//                     is_minima = false;
//                     break;
//                 }
//                 else if(decp_data[j] == decp_data[i] and j<i){
//                     is_minima = false;
//                     break;
//                 }
//             }
        
//     if(((is_maxima && (or_direction_as[i]!=-1)) or (!is_maxima && (or_direction_as[i]==-1)))){
//         // if(lowGradientIndices[i] == 0){
//         //     cout<<or_direction_as[i]<<de_direction_as[i]<<endl;
//         // }
//         // if(lowGradientIndices[i] == 0){
//             int idx_fp_max = atomic_fetch_add_explicit(&count_f_max, 1, memory_order_relaxed);
//             // if(i==25026){cout<<or_direction_as[i]<<is_maxima<<endl;}
//             all_max[idx_fp_max] = i;
//         // }
        
//     }

//     else if (((is_minima && (or_direction_ds[i]!=-1)) or (!is_minima && or_direction_ds[i]==-1)) and lowGradientIndices[i] == 0) {
//         int idx_fp_min = atomic_fetch_add_explicit(&count_f_min, 1, memory_order_relaxed);// in one instruction
//         all_min[idx_fp_min] = i;
    
//     }     
        
// }
    

// }
// extern "C" {
//     int mainCuda(std::vector<int> a, std::vector<int> b);
// }
// fix_process
// extern __global__ void init_or_data1(int numElements);
// extern void init_or_data(std::vector<int> *a, std::vector<int> *b,std::vector<int> *c, std::vector<int> *d, std::vector<double> *input_data,std::vector<double> *decp_data, int num);
// extern int fix_process(std::vector<double>& decp_data);
extern void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,std::vector<int> *dec_label1,std::vector<int> *or_label1,int width1, int height1, int depth1, std::vector<int> *low,double bound1,float &datatransfer,float &finddirection,double &right);
// extern void update_de_direction(std::vector<int> *c,std::vector<int> *d);
extern void fix_process(std::vector<int> *c,std::vector<int> *d, std::vector<double> *decp_data1, float &datatransfer, float &finddirection, float &getfcp, float &fixtime_cp,int &cpite);
extern void mappath1(std::vector<int> *label, std::vector<int> *direction_as, std::vector<int> *direction_ds, float &finddirection, float &mappath_path, float &datatransfer,int type=0);

void getlabel(int i){
    
    
    
    int cur = dec_label[i*2+1];
    int next_vertex;
    // cout<<cur<<endl;
    if (cur==-1){
        
        return;
    }
    else if (de_direction_as[cur]!=-1){
        
        // cout<<cur<<" "<<de_direction_as[cur]<<endl;
        int direc = de_direction_as[cur];
        int row = cur/width;
        int rank1 = cur%width;
        
        switch (direc) {
            case 1:
                next_vertex = (row)*width + (rank1-1);
                break;
            case 2:
                next_vertex = (row-1)*width + (rank1);
                break;
            case 3:
                next_vertex = (row-1)*width + (rank1+1);
                break;
            case 4:
                next_vertex = (row)*width + (rank1+1);
                break;
            case 5:
                next_vertex = (row+1)*width + (rank1);
                break;
            case 6:
                next_vertex = (row+1)*width + (rank1-1);
                break;
        };

        cur = next_vertex;
        
        if (de_direction_as[cur] != -1){
            
            un_sign_as+=1;
        }

        if(de_direction_as[i]!=-1){
            dec_label[i*2+1] = cur;
            
        }
        else{
            dec_label[i*2+1] = -1;
        };
        
    }

    
    
    cur = dec_label[i*2];
    int next_vertex1;
    if(cur==-1){
        return;
    }
    if (de_direction_as[cur]!=-1){
        // printf("%d\n", cur);
        int direc = de_direction_ds[cur];
        int row = cur/width;
        int rank1 = cur%width;
        
        switch (direc) {
            case 1:
                next_vertex1 = (row)*width + (rank1-1);
                break;
            case 2:
                next_vertex1 = (row-1)*width + (rank1);
                break;
            case 3:
                next_vertex1 = (row-1)*width + (rank1+1);
                break;
            case 4:
                next_vertex1 = (row)*width + (rank1+1);
                break;
            case 5:
                next_vertex1 = (row+1)*width + (rank1);
                break;
            case 6:
                next_vertex1 = (row+1)*width + (rank1-1);
                break;
        };

        cur = next_vertex1;
        
        if (de_direction_ds[cur] != -1){
            un_sign_ds+=1;
            // printf("%d \n",i);
        }

        if(de_direction_ds[i]!=-1){
            dec_label[i*2] = cur;
            
        }
        else{
            dec_label[i*2] = -1;
        };
        
    }

    

}
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
    omp_set_num_threads(44);
    std::string dimension = argv[1];
    double range = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int mode = std::stoi(argv[4]);
    double right_labeled_ratio;
    double target_br;
    float datatransfer = 0.0;
    float mappath_path = 0.0;
    float getfpath = 0.0;
    float fixtime_path = 0.0;
    float finddirection = 0.0;
    float getfcp = 0.0;
    float fixtime_cp = 0.0;
    if(mode==1){
        target_br = std::stod(argv[5]);
    }
    std::istringstream iss(dimension);
    char delimiter;
    std::string filename;
    if (std::getline(iss, filename, ',')) {
        // 接下来读取整数值
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
    
    

    
    
    // 显式指定模板参数为 double
    // std::vector<double> dataDouble = getdata2<double>("datafile.txt");
    auto start = std::chrono::high_resolution_clock::now();
    input_data = getdata2(inputfilename);
    auto min_it = std::min_element(input_data.begin(), input_data.end());
    auto max_it = std::max_element(input_data.begin(), input_data.end());
    double minValue = *min_it;
    double maxValue = *max_it;
    bound = (maxValue-minValue)*range;
    
    // exit(0);
    std::ostringstream stream;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    // 设置精度为最大可能的双精度浮点数有效数字
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << std::defaultfloat << bound;  // 使用默认的浮点表示
    std::string valueStr = stream.str();
    size2 = input_data.size();
    std::string cpfilename = "compressed_data/compressed_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".sz";
    std::string decpfilename = "decompressed_data/decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string fix_path = "decompressed_data/fixed_decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string command;
    cout<<decpfilename<<endl;
    cout<<bound<<", "<<std::to_string(bound)<<endl;
    // exit(0);
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
    auto ends = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = ends - start;
    double compression_time = duration.count();
    std::vector<double> decp_data_copy(decp_data);
    
    or_direction_as.resize(size2);
    or_direction_ds.resize(size2);
    de_direction_as.resize(size2);
    de_direction_ds.resize(size2);
    or_label.resize(size2*2, -1);
    dec_label.resize(size2*2, -1);
    std::vector<std::vector<double>> time_counter;
    lowGradientIndices = find_low();
    lowGradientIndices.resize(size2, 0);
    
    adjacency = _compute_adjacency();
    
    all_max.resize(size2);
    all_min.resize(size2);
    
    std::random_device rd;  // 随机数生成器
    std::mt19937 gen(rd()); // 以随机数种子初始化生成器
    
    
    

    int cnt=0;
 
    
    
    
    
    
    
    
    // #pragma omp parallel for
    // for (int i=0;i<size2;++i){
    //     // if(i%100000==0) cout<<i/100000<<endl;
    //     if(lowGradientIndices[i] == 1){
    //         continue;
    //     }
    //     // if (or_mesh.maxi.find(i) == or_mesh.maxi.end()){
            
    //         or_direction_as[i] = find_direction(i,input_data,0);
    //     // }
    //     // else{
            
    //     //     or_direction_as[i]= -1;
    //     // };
        
        
    //     // if (or_mesh.mini.find(i) == or_mesh.mini.end()){
            
    //         or_direction_ds[i] = find_direction(i,input_data,1);
            
    //     // }
    //     // else{
            
    //     //     or_direction_ds[i] = -1;
    //     // };
        
        
       
        

    //  };
    // // exit(0);
    // // cout<<"dahk"<<endl;

    
    
    
    // #pragma omp parallel for
    
    // for (int i=0;i<size2;++i){
    //             if(lowGradientIndices[i] == 1){
    //             continue;
    //         }
    //         de_direction_as[i] = find_direction(i,decp_data,0);
        
    //         de_direction_ds[i] = find_direction(i,decp_data,1);
        
    // };
    // cout<<"de完成"<<endl;
    
    


    
    // or_label = mappath(or_mesh,or_direction_as,or_direction_ds);
    // cout<<lowGradientIndices.size()<<endl;
    // cout<<"map完成"<<endl;
    // exit(0);
    // std::ofstream file1("or"+std::to_string(width)+"_label.txt");
    // std::string result = "";
    // for (const auto& p : or_label) {
    //     result+="[";
    //     result+=std::to_string(p.first)+","+std::to_string(p.second);
    //     result+="],";
    // }
    // file1 << result << "\n";
    // file1 << "#"<<",";
    
    // result = "[";
    // for( int i:or_mesh.maxi){
    //     result+=std::to_string(i);
    //     result+=",";
    // }
    // result+="]";
    // file1 << result;
    // file1 << "\n";
    // file1 << "#"<<",";
    // result = "[";
    // for( int i:or_mesh.mini){
    //     result+=std::to_string(i);
    //     result+=",";
    // }
    // result+="]";
    // file1 << result;

    // // // 关闭文件
    // file1.close();
    
    // std::ofstream file1("or_label.txt");

    // // 写入数据
    
    // for (const auto& p : or_label) {
        
    //     file1 << p.first << ", " << p.second << "\n";
    // }
    
    
    // file1 << "#"<<",";
    // for( int i:or_mesh.maxi){
    //     file1 << i << ",";
    // }
    // file1 << "\n";
    // file1 << "#"<<",";
    // for( int i:or_mesh.mini){
    //     file1 << i << ",";
    // }
    // // 关闭文件
    // file1.close();
    
    
    
    
    // dec_label = mappath(de_mesh,de_direction_as,de_direction_ds);
    
    
    
    
    
    
    
    // while (count_f_max>0 or count_f_min>0){
    //     cpite +=1;
    //     start5 = std::chrono::high_resolution_clock::now();
        
    //     cout<<"修复cp: "<<count_f_max<<", "<<count_f_min<<endl;
    //     // cout<<"修复:"<<or_direction_as[25026]<<", "<<de_direction_as[25026]<<endl;
    //     // if(count_f_max ==3){
    //     //     for(int i=0;i<count_f_max;i++){
    //     //         cout<<all_max[i]<<endl;
                
    //     //         for(int i:adjacency[i]){
    //     //             cout<<i<<", "<<input_data[i]<<endl;
    //     //         }
    //     //     }
    //     //     cout<<endl;
    //     // }
    //     // if(count_f_min == 84){
    //     //     for(int i=0;i<count_f_min;i++){
    //     //         cout<<all_min[i]<<"," ;
    //     //     }
    //     //     cout<<endl;
    //     // }
        
    //     start4 = std::chrono::high_resolution_clock::now();
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
        
        
       
        
    //     end4 = std::chrono::high_resolution_clock::now();
    //     d = end4-start4;
    //     fixtime_cp += d.count();
        
    //     start4 = std::chrono::high_resolution_clock::now();
    //     #pragma omp parallel for
    //     for (int i=0;i<size2;++i){
            
    //             if(lowGradientIndices[i] == 1){
    //                 continue;
    //             }
    //             de_direction_as[i] = find_direction(i,decp_data,0);
            
    //             de_direction_ds[i] = find_direction(i,decp_data,1);
            
            
    //     };
    //     // get_wrong_index_cp();
    //     get_false_criticle_points();
    //     end4 = std::chrono::high_resolution_clock::now();
    //     d = end4-start4;
    //     getfcp += d.count();
        

    //     d = end4-start5;
    //     total+= d.count();
        
    //     // cout<<"time for fix:"<<temp<<endl;
    
    //     // cout<<"time for assign value:"<<temp2<<endl;
    //     // cout<<"time for get cp:"<<temp3<<endl;
    //     // cout<<"time for get false cp:"<<temp5<<endl;
    //     // cout<<"total time："<<total<<endl;
    //     // cout<<"time for fix"<<temp/total<<endl;
        
    //     // cout<<"assign value"<<temp2/total<<endl;
    //     // cout<<"get cp"<<temp3/total<<endl;
    //     // cout<<"get direction"<<temp1/total<<endl;
    //     // cout<<"get false cp"<<temp5/total<<endl;
    //     // cout<<"错误cp:"<<count_max<<","<<count_min<<endl;
    //     // cout<<cpite<<endl;
    //     // exit(0);
    // }
    
    // cout<<"开始"<<endl;
    // init_or_data(&or_direction_as, &or_direction_ds, &de_direction_as, &de_direction_ds, &input_data, &decp_data,size2);
    std::vector<int>* dev_a = &or_direction_as;
    std::vector<int>* dev_b = &or_direction_ds;
    std::vector<int>* dev_c = &de_direction_as;
    std::vector<int>* dev_d = &de_direction_ds;
    std::vector<double>* dev_e = &input_data;
    std::vector<double>* dev_f = &decp_data;
    std::vector<int>* dev_g = &lowGradientIndices;
    
    // init_or_data(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f,size2);
    std::vector<int>* dev_q = &dec_label;
    std::vector<int>* dev_m = &or_label;
    // cout<<"开始"<<endl;
    
    start = std::chrono::high_resolution_clock::now();
    
    double compressed_dataSize = fs::file_size(cpfilename);
    double br = (compressed_dataSize*8)/size2;
    start = std::chrono::high_resolution_clock::now();
    init_inputdata(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m,dev_q,width, height, depth, dev_g, bound,datatransfer,finddirection, right_labeled_ratio);
    ends = std::chrono::high_resolution_clock::now();
    duration = ends - start;
    double additional_time = duration.count();
    std::cout<<"compression_time: "<<compression_time<<std::endl;
    std::cout<<"additional_time: "<<additional_time<<std::endl;
    std::ofstream outFilep("result/compression_time.txt", std::ios::app);
    // 检查文件是否成功打开
    if (!outFilep) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1; // 返回错误码
    }

    outFilep << filename + "_" +compressor_id+"_" + std::to_string(range) + ":" << "compression_time:" <<compression_time<<" additional_time:" <<additional_time <<"ratio:"<<additional_time/(additional_time+compression_time)<< std::endl;
    // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
    // outFilep << std::to_string(number_of_thread)<<":" << std::endl;
    
    outFilep << "\n"<< std::endl;
    outFilep.close();
    
    // return 0;
    // std::ofstream outFile11("max_label"+filename+std::to_string(bound)+".txt");

    // if (outFile11.is_open()) {
    //     outFile11 << std::setprecision(std::numeric_limits<double>::max_digits10);

    //     for (int i=0;i<size2;i++) {

    //         if(or_label[i*2]==-1){
    //             outFile11 << i << std::endl;
    //         }
    //         else if(or_label[i*2]==i){
    //             outFile11 << -1 << std::endl;
    //         }
    //         else{
    //             outFile11 << or_label[i*2] << std::endl;
    //         }
            
    //         if(or_label[i*2+1]==-1){
    //             outFile11 << i << std::endl;
    //         }
    //         else if(or_label[i*2+1]==i){
    //             outFile11 << -1 << std::endl;
    //         }
    //         else{
    //             outFile11 << or_label[i*2+1] << std::endl;
    //         }

            
            
    //     }
    //     outFile11.close();
    // } else {
    //     std::cerr << "Unable to open the file for writing." << std::endl;
    // }
    
    
    // // std::chrono::duration<double> durationt = endt - start;
    // cout<<"duration: "<<duration.count()<<endl;
    // cout<<"find_direction:"<<finddirection*1000<<endl;
    // cout<<"getfcp:"<<getfcp<<endl;
    // cout<<"mappath_path:"<<mappath_path<<endl;
    // cout<<"getfpath:"<<getfpath<<endl;
    // cout<<"fixfcp:"<<fixtime_cp<<endl;
    // cout<<"fixpath:"<<fixtime_path<<endl;
    // exit(0);
    
    
    // cout<<"wancheng"<<endl;
    // update_de_direction(dev_c,dev_d);
    // for(int i=0;i<input_data.size();i++){
    //     if(abs(input_data[i]-decp_data[i])>bound){
    //         cout<<abs(input_data[i]-decp_data[i])<<endl;
    //     }
    // }
    // exit(0);
    // cout<<"djal"<<endl;
    // 10412175,6513917
    // 10117038,5735947
    
    
        
    
    // #pragma omp parallel for
    // for (int i=0;i<size2;++i){
    //     if(lowGradientIndices[i] == 1){
    //         continue;
    //     }
    //         de_direction_as[i] = find_direction(i,decp_data,0);
        
    //         de_direction_ds[i] = find_direction(i,decp_data,1);
        
    // };
    // for(int index=0;index<size2;index++){
    //     if(de_direction_ds[index]!=-1){
    //         dec_label[index*2] = index;
    //     }
    //     else{
    //         dec_label[index*2] = -1;
    //     }

    //     if(de_direction_as[index]!=-1){
    //         dec_label[index*2+1] = index;
    //     }
    //     else{
    //         dec_label[index*2+1] = -1;
    //     }
    // }
    
    // while(un_sign_as>0 or un_sign_ds>0){
    //     un_sign_as=0;
    //     un_sign_ds=0;
    //     for(int i=0;i<size2;i++){
    //         // cout<<i<<endl;
    //         getlabel(i);
    //     }
    //     // cout<<un_sign_ds<<","<<un_sign_as<<endl;
        
    // }
    // for(int i=1;i<size2*2;i+=2){
    //     int label1 = dec_label[i];
    //     if(de_direction_as[label1]!=-1 and label1!=-1){
    //         cout<<i<<", "<<label1<<", "<<de_direction_as[label1]<<endl;
    //     }
    // }

    
    
    
    // for(int i=0;i<size2;i++){
    //     if(de_direction_as[i]==-1){
    //         cout<<i<<endl;
    //     }
    // }
    // exit(0);
    
    // std::ofstream outFile(fix_path);
    // // std::ofstream outFile("values.bin", std::ios::binary | std::ios::out);
    // if (outFile.is_open()) {
    //     outFile.write(reinterpret_cast<const char*>(decp_data.data()), decp_data.size() * sizeof(double));
    // }
    // outFile.close();
    // start = std::chrono::high_resolution_clock::now();
    std::vector<double> indexs;
    std::vector<double> edits;
    for (int i=0;i<input_data.size();i++){
        
        if (decp_data_copy[i]!=decp_data[i]){
            indexs.push_back(i);
            edits.push_back(decp_data[i]-decp_data_copy[i]);
            cnt++;
        }
    }
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
    double ratio = double(cnt)/(decp_data_copy.size());
    cout<<cnt<<","<<ratio<<endl;
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
    
    // cout<<compressed_indexSize<<", "<<original_indexSize<<", "<<original_indexSize/compressed_indexSize<<endl;
    // cout<<compressed_editSize<<", "<<original_editSize<<", "<<original_editSize/compressed_editSize<<endl;
    // cout<<compressed_dataSize<<", "<<original_dataSize<<", "<<original_dataSize/compressed_dataSize<<endl;
    double overall_ratio = (original_indexSize+original_editSize+original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    double bitRate = 64/overall_ratio; 
    std::cout << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
    std::cout << std::setprecision(17)<<"CR: "<<original_dataSize/compressed_dataSize << std::endl;
    std::cout << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
    std::cout << std::setprecision(17)<<"BR: "<< (compressed_dataSize*8)/size2 << std::endl;
    
    outFilep <<  "original_compressed_size:" <<compressed_dataSize<< " additional_size:" <<compressed_editSize+compressed_indexSize<<" ratio:"<<(compressed_editSize+compressed_indexSize)/compressed_dataSize<< std::endl;
    outFilep<< std::setprecision(17)<< "OCR: "<<overall_ratio << " CR: "<<original_dataSize/compressed_dataSize << " OBR: "<<bitRate << " BR: "<< (compressed_dataSize*8)/size2 << std::endl;
    // finddirection:0, getfcp:1,  mappath2, fixcp:3
    
    // outFilep << std::to_string(number_of_thread)<<":" << std::endl;
    
    outFilep << "\n"<< std::endl;
    outFilep.close();
    exit(0);
    
    


    double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);
    cout<<psnr<<", "<<fixed_psnr<<endl;
    // cout<<"right: "<<right_labeled_ratio<<endl;
    cout<<"relative range: "<<range<<endl;
    
    std::ofstream outFile3("./result/result_"+filename+"_"+compressor_id+"tmp3_detailed.txt", std::ios::app);

    // 检查文件是否成功打开
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1; // 返回错误码
    }

    
    outFile3 << std::to_string(bound)<<":" << std::endl;
    outFile3 << std::setprecision(17)<< "related_error: "<<range << std::endl;
    outFile3 << std::setprecision(17)<< "OCR: "<<overall_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"CR: "<<original_dataSize/compressed_dataSize << std::endl;
    outFile3 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
    outFile3 << std::setprecision(17)<<"BR: "<< (compressed_dataSize*8)/size2 << std::endl;
    outFile3 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
    outFile3 << std::setprecision(17)<<"fixed_psnr: "<<fixed_psnr << std::endl;

    outFile3 << std::setprecision(17)<<"right_labeled_ratio: "<<1-right_labeled_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << std::setprecision(17)<<"relative: "<<range << std::endl;
    outFile3 << "\n" << std::endl;
    // 关闭文件
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;

    cout<<overall_ratio * bitRate<<endl;
    cout<<overall_ratio<<","<<bitRate<<endl;

    std::string rm_command = "rm -f " + fix_path;

    // 执行删除文件的命令
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    rm_command = "rm -f " + cpfilename;
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    rm_command = "rm -f " + decpfilename;
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    rm_command = "rm -f " + indexfilename;
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    rm_command = "rm -f " + editsfilename;
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    rm_command = "rm -f " + compressedindex;
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    rm_command = "rm -f " + compressededits;
    result = system(rm_command.c_str());

    // 检查删除操作的结果
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cerr << "Error deleting file." << std::endl;
    }

    // std::string indexfilename = "data"+filename+".bin";
    // std::string editsfilename = "data_edits"+filename+".bin";
    // std::string compressedindex = "data"+filename+".bin.zst";
    // std::string compressededits = "data_edits"+filename+".bin.zst";

        // 尝试删除文件
    

    
    // std::vector<double> diff(input_data.size());

    // // 计算差值的绝对值
    // std::transform(input_data.begin(), input_data.end(), decp_data.begin(), diff.begin(),
    //                [](double a, double b) {
    //                    return std::abs(a - b);
    //                });

    // 找出最大的绝对差值
    // auto max_diff = *std::max_element(diff.begin(), diff.end());

    // std::cout << "The largest absolute difference is: " << max_diff << std::endl;
    // std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
    // cout<<cnt<<","<<ratio<<endl;
    
    // std::cout << "get_fcp: " << getfcp << " seconds" << std::endl;
    // std::cout << "fixtime_cp: " << fixtime_cp << " seconds" << std::endl;
    // std::cout << "fixtime_path: " << fixtime_path << " seconds" << std::endl;
    // std::cout << "searchtime: " << searchtime << " seconds" << std::endl;
    // std::cout << "finddirection: " << searchdirection_time << " seconds" << std::endl;
    // std::cout << "getfpath:" << get_path << std::endl;
    // std::cout << "iteration number:" << ite << std::endl;
    
    // std::string result = "[";
    // for(int i=0;i<record1.size();i++){
    //     std::string result1 = "[";
    //     std::vector<float> temp = record1[i];
    //     for(auto k:temp){
    //         result1+=std::to_string(k);
    //         result1+=",";
    //     }
    //     result1+="]";
    //     result1+=",";
    //     result+=result1;
    // }
    // result+="]";
    // std::cout<<result<<std::endl;
    // result = "[";
    // for(int i=0;i<record_ratio.size();i++){
    //     std::string result1 = "[";
    //     std::vector<float> temp = record_ratio[i];
    //     for(auto k:temp){
    //         result1+=std::to_string(k);
    //         result1+=",";
    //     }
    //     result1+="]";
    //     result1+=",";
    //     result+=result1;
    // }
    // result+="]";
    // std::cout<<result<<std::endl;
    
    // std::system("rm -f "+decpfilename);
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
            outFile5 << std::setprecision(17)<<"relative_error: "<<range << std::endl;
            outFile5 << std::setprecision(17)<<"OCR: "<<overall_ratio << std::endl;
            outFile5 << std::setprecision(17)<<"CR: "<<original_dataSize/compressed_dataSize << std::endl;
            outFile5 << std::setprecision(17)<<"OBR: "<<bitRate << std::endl;
            outFile5 << std::setprecision(17)<<"BR: "<< (compressed_dataSize*8)/size2 << std::endl;
            outFile5 << std::setprecision(17)<<"psnr: "<<psnr << std::endl;
            outFile5 <<std::setprecision(17)<< "fixed_psnr: "<<fixed_psnr << std::endl;

            //outFile5 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
            outFile5 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
            outFile5 << "\n" << std::endl;
            // 关闭文件
            outFile5.close();
            std::cout << "target_bitrate_founded" << std::endl;
        }
    }
    return 0;
}