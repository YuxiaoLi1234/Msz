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

using namespace std;
namespace fs = std::filesystem;

// 构建mesh
// g++ -std=c++17 -O3 -g hello.cpp -o helloworld
// g++ -std=c++17 -O3 -g -fopenmp -c preserve2d.cpp -o hello2d.o
// g++-12 -fopenmp hello2d.o kernel2d.o -lcudart -o helloworld3d
// g++-12 -fopenmp -std=c++17 -O3 -g hello2.cpp -o helloworld2
// g++ -fopenmp hello2d.o kernel2d.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld2d
int width;
int height;
int size2;
int un_sign_as;
int un_sign_ds;
int ite = 0;
std::vector<int> all_max, all_min, all_d_max, all_d_min;
atomic_int count_max = 0;
atomic_int count_min = 0;
atomic_int count_f_max = 0;
atomic_int count_f_min = 0;

std::vector<double> input_data;

std::vector<int> record;
std::vector<std::vector<float>> record1;
std::vector<std::vector<float>> record_ratio;
int directions1[12] =  {0, -1, -1, 0, -1, 1, 0, 1, 1, 0, 1, -1};

int direction_to_index_mapping[6][2] = {
    
    {0, -1},   
    {-1, 0},   
    {-1, 1},  
    {0, 1},    
    {1, 0},    
    {1, -1}   
};
std::unordered_map<int, double> maxrecord;
std::unordered_map<int, double> minrecord;
double bound;
std::vector<int> findUniqueElements(const std::unordered_set<int>& set1, const std::unordered_set<int>& set2) {
    std::vector<int> uniqueElements;
    // 检查 set1 中有而 set2 中没有的元素
    for (const auto& elem : set1) {
        if (set2.find(elem) == set2.end()) {
            uniqueElements.push_back(elem);
        }
    }

    // 检查 set2 中有而 set1 中没有的元素
    for (const auto& elem : set2) {
        if (set1.find(elem) == set1.end()) {
            uniqueElements.push_back(elem);
        }
    }

    return uniqueElements;
}
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
std::vector<int> find_low(){
    std::vector<int> lowGradientIndices(size2, 0);
    
    const double threshold = 1e-16; // 梯度阈值
    // 遍历三维数据计算梯度
    for (int i = 0; i < width; ++i) {
        
        for (int j = 0; j < height; ++j) {
            
           
                
                int rm = i * height + j;
                
                
                // for(int q=0;q<12;q++){
                for (int d = 0; d < 6; d++) {
            
                    int dirX = directions1[d * 2];     
                    int dirY = directions1[d * 2 + 1]; 
                    int newX = i + dirX;
                    int newY = j + dirY;
                    int r = newX * height + newY;
                    if(r>=0 and r<size2){
                        double gradZ3 = abs(input_data[r] - input_data[rm]) / 2.0;
                        if (gradZ3<=threshold) {
                            lowGradientIndices[rm]=1;
                            lowGradientIndices[r]=1;
                        }
                    // }
                }
                }
            
        }
    }
    // cout<<"ok"<<endl;
    // int count = std::count(lowGradientIndices.begin(), lowGradientIndices.end(), 1);

    // std::cout << "The number of 1s in the vector is: " << count << std::endl;
    return lowGradientIndices;
}

std::vector<int> lowGradientIndices;
std::vector<std::vector<int>> _compute_adjacency(){
    std::vector<std::vector<int>> adjacency;
    for (int i = 0; i < size2; ++i) {
            int x = i / height; // Get the x coordinate
            int y = i % height; // Get the y coordinate
            std::vector<int> adjacency_temp;
            
            for (int d = 0; d < 6; d++) {
            
                int dirX = directions1[d * 2];     
                int dirY = directions1[d * 2 + 1]; 
                
                int newX = x + dirX;
                int newY = y + dirY;
                int r = newX * height + newY; // Calculate the index of the adjacent vertex

                // Check if the new coordinates are within the bounds of the mesh
                if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height && lowGradientIndices[r] != 1) {
                    adjacency_temp.push_back(r);
                }
            }
            // cout<<adjacency_temp.size()<<endl;
            
            adjacency.push_back(adjacency_temp);
        }
        // exit(0);
    return adjacency;
}
std::vector<std::vector<int>> adjacency;

std::vector<double> add_noise(const std::vector<double>& data, double x) {
    std::vector<double> noisy_data = data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-x, x);

    std::for_each(noisy_data.begin(), noisy_data.end(), [&dis, &gen](double &d){
        d += dis(gen);
    });

    return noisy_data;
};

std::map<std::pair<int, int>, int> createDirectionMapping() {
    std::map<std::pair<int, int>, int> direction_mapping;
    direction_mapping[std::make_pair(0,-1)] = 1;
    direction_mapping[std::make_pair(-1,0)] = 2;
    direction_mapping[std::make_pair(-1,1)] = 3;
    direction_mapping[std::make_pair(0,1)] = 4;
    direction_mapping[std::make_pair(1,0)] = 5;
    direction_mapping[std::make_pair(1,-1)] = 6;

    return direction_mapping;
};

std::map<int, std::pair<int, int>> createReverseMapping(const std::map<std::pair<int, int>, int>& originalMap) {
    std::map<int, std::pair<int, int>> reverseMap;
    for (const auto& pair : originalMap) {
        reverseMap[pair.second] = pair.first;
    }
    return reverseMap;
}
std::map<std::pair<int, int>, int> direction_mapping = createDirectionMapping();

std::map<int, std::pair<int, int>> reverse_direction_mapping = createReverseMapping(direction_mapping);
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
                if(this->values[j] == this->values[i] and j>i){
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
                if(this->values[j] == this->values[i] and j<i){
                    is_minima = false;
                    break;
                }
            }

            if (is_maxima and lowGradientIndices[i] == 0) {
                local_maxi_temp.push_back(i);
            }
            if (is_minima  and lowGradientIndices[i] == 0) {
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



std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> linspaced;

    if (num == 0) { 
        return linspaced; 
    }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); 
    
    // 确保end被包括进去
    return linspaced;
}

std::vector<double> getdata(std::string filename){
     std::vector<double> data;
     std::string line;
     std::ifstream file(filename);

     if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return data;
     }

     // 读取每一行
     
     while (std::getline(file, line)) {
          std::vector<double> row;
          std::stringstream ss(line);
          std::string value;

          // 分割行中的每个值
          
          while (std::getline(ss, value, ',')) {
               data.push_back(std::stod(value)); // 将字符串转换为 double
            //    if(cnt1==7319){
                
            //    }
              
          }
        //   data.push_back(row);
     }
    //  cout<<data[7391]<<endl;
     file.close();
     
     return data;
};
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
std::string inputfilename;
std::string decompfilename;



// std::vector<int> ordered_input_data = processdata(input_data);

std::vector<double> decp_data;

// std::vector<double> decp_data = add_noise(input_data,bound);

// std::vector<int> ordered_decp_data = processdata(decp_data);

myTriangularMesh or_mesh;
myTriangularMesh de_mesh;
std::vector<int> wrong_maxi_cp;
std::vector<int> wrong_min_cp;
std::vector<int> wrong_index_as;
std::vector<int> wrong_index_ds;


int getDirection(const std::map<std::pair<int, int>, int>& direction_mapping, int row_diff, int col_diff) {
    auto it = direction_mapping.find(std::make_pair(row_diff, col_diff));
    
    if (it != direction_mapping.end()) {
        return it->second;
    } else {
        return -1; 
    }
}
int from_direction_to_index(int cur, int direc){
    if (direc==-1) return cur;
    int row = cur / height;
    int rank1 = cur % height;
    // printf("%d %d\n", row, rank1);
    if (direc > 0 && direc <= 6) {
        int delta_row = direction_to_index_mapping[direc-1][0];
        int delta_col = direction_to_index_mapping[direc-1][1];
        int next_row = row + delta_row;
        int next_col = rank1 + delta_col;
        // printf("%d \n", next_row * width + next_col);
        return next_row * height + next_col;
    }
    else {
        return -1;
    }
    
};
int find_direction (int index,  std::vector<double> &data ,int direction){
    
   
    double mini = 0;
    int size1 = adjacency[index].size();
    
    std::vector<int> indexs = adjacency[index];
    int largetst_index = index;
    
    if (direction == 0){
        
        for(int i =0;i<size1;++i){
            
            if(data[indexs[i]]-data[index]>mini or (data[indexs[i]]-data[index]==mini and indexs[i]>largetst_index)){
                mini = data[indexs[i]]-data[index];
                
                largetst_index = indexs[i];
                // }
                
            };
        };
        
        
    }
    else{
        for(int i =0;i<size1;++i){
            if(data[indexs[i]]-data[index]<mini or (data[indexs[i]]-data[index]==mini and indexs[i]<largetst_index)){
                mini = data[indexs[i]]-data[index];
                
                largetst_index = indexs[i];

                
            };
        };
    };
    
    int row_l = largetst_index / width;
    int row_i = index / width;
    
    int row_diff = row_l - row_i;
    int col_diff = (largetst_index % width) - (index % width);
    
    int d = getDirection(direction_mapping, row_diff, col_diff);
    
    
    return d;

};
int find_direction1 (int index,std::map<int,int> &data ,int direction){

    double mini = 0;
    int size1 = adjacency[index].size();
    std::vector<int> indexs = adjacency[index];
    int largetst_index;
    
    if (direction == 0){
        
        for(int i =0;i<size1;++i){
            
            if(data[indexs[i]]-data[index]>mini){
                mini = data[indexs[i]]-data[index];
                largetst_index = indexs[i];
            };
        };
        
        
    }
    else{
        for(int i =0;i<size1;++i){
            if(data[adjacency[index][i]]-data[index]<mini){
                mini = data[adjacency[index][i]]-data[index];
                largetst_index = adjacency[index][i];
            };
        };
    };
    // cout<<largetst_index<<","<<index<<endl;
    int row_l = largetst_index / width;
    int row_i = index / width;
    
    int row_diff = row_l - row_i;
    int col_diff = (largetst_index % width) - (index % width);
    
    int d = getDirection(direction_mapping, row_diff, col_diff);
    
    
    return d;

};
std::vector<int> mappath(myTriangularMesh& mesh, std::vector<int> direction_as, std::vector<int> direction_ds){
    // std::vector<std::pair<int, int>> label(size2, std::make_pair(-1, -1));
    std::vector<int> label(size2*2, -1);

    #pragma omp parallel for
    for (int i = 0;i<size2;++i){
        int cur = i;
        
        while (direction_as[cur]!=-1){
            
            int direc = direction_as[cur];
            
            int row = cur/width;
            int rank1 = cur%width;
            
            int next_vertex;
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
            
            if (label[cur*2+1] != -1){
                cur = label[cur*2+1];
                break;
            }
            
        }

        if(direction_as[i]!=-1){
            label[i*2+1] = cur;
        }
        else{
            label[i*2+1] = -1;
        };
        
        cur = i;

        while (direction_ds[cur]!=-1){
            int direc = direction_ds[cur];
            int row = cur/width;
            int rank1 = cur%width;
            int next_vertex;
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

            if (label[cur*2+0] != -1){
                cur = label[cur*2+0];
                break;
            }
        }

        if(direction_ds[i]!=-1){
            label[i*2+0] = cur;
        }
        else{
            label[i*2+0] = -1;
        };

    };
    
    return label;
};
std::vector<int> or_direction_as;
std::vector<int> or_direction_ds;
std::vector<int> de_direction_as;
std::vector<int> de_direction_ds;

// std::vector<std::pair<int, int>> or_label(size2, std::make_pair(-1, -1));
std::vector<int> dec_label;
std::vector<int> or_label;
// std::vector<std::pair<int, int>> dec_label(size2, std::make_pair(-1, -1));
bool isNumberInArray(const std::vector<int>& array, int number) {
    for (int num : array) {
        if (num == number) {
            return true;  // 找到了，返回 true
        }
    }
    return false;  // 遍历完毕，未找到，返回 false
}

int fix_maxi_critical(int index, int direction){
    
    if (direction == 0){
        
        if (or_direction_as[index]!=-1){
            
            int next_vertex = from_direction_to_index(index,or_direction_as[index]);
            int smallest_vertex = next_vertex;
            double threshold = std::numeric_limits<double>::lowest();
            
            
            for(int i:adjacency[index]){
                if(input_data[i]<input_data[index] and input_data[i]>threshold and i!=next_vertex){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d = (bound - (input_data[index]-decp_data[index]))/2.0;
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                
                de_direction_as[index]=or_direction_as[index];
            
                return 0;
            }
            
            if(d>=1e-16){
                
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
                        
                        if (decp_data[index]>=threshold and threshold<=decp_data[next_vertex]){
                            
                            while(decp_data[index] - d < threshold and d>=2e-16)
                            {
                                d/=2;
                            }
                            
                            
                        }
                        else if(threshold>decp_data[next_vertex]){
                            
                            
                            double diff2 = (bound-(input_data[smallest_vertex]-decp_data[smallest_vertex]))/2;
                            
                            if(diff2>1e-16){
                                while(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)>bound and diff2>=2e-16 and decp_data[smallest_vertex]-diff2>decp_data[next_vertex]){
                                    
                                    diff2/=2;
                                }
                                
                                if(abs(input_data[smallest_vertex]-decp_data[smallest_vertex]+diff2)<=bound){
                                    decp_data[smallest_vertex]-=diff2;
                                    // if(index==97) cout<<"处理97的时候: "<<decp_data[next_vertex]<<", "<<decp_data[index]<<endl;
                                }
                                
                                
                            }
                            
                        }

                        if(abs(input_data[index]-(decp_data[index]-d))<=bound and decp_data[index]>=decp_data[next_vertex]){
                            decp_data[index] -= d;
                        }
                        
                        
                   
                };
                     }

                 
            
                
            }
            else{
                
                if(decp_data[index]>=decp_data[next_vertex]){
                    if(abs(input_data[index]-(input_data[next_vertex] -bound+ decp_data[index])/2.0)<=bound){
                        decp_data[index] = (input_data[next_vertex] -bound + decp_data[index])/2.0;
                    }
                    else{
                        
                        decp_data[index] = input_data[index] - bound;
                    }
                    
                }
                
            }
            
            
        
        }
        else{
             
            int largest_index = from_direction_to_index(index,de_direction_as[index]);
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                de_direction_as[index] = -1;
            }
            if(d>=1e-16){
                
                if (decp_data[index]<=decp_data[largest_index]){
                    if(abs(input_data[largest_index]-decp_data[index]+d)){
                        decp_data[largest_index] = decp_data[index]-d;
                    }
                }
                
            
                
            }
            
            else{
                if(decp_data[index]<=decp_data[largest_index]){
                    decp_data[index] = input_data[index] + bound;
                }
                    
            }
            
        }
        
        
    
    }
    
    else{
        
        if (or_direction_ds[index]!=-1){
            int next_vertex= from_direction_to_index(index,or_direction_ds[index]);
            
            double diff = (bound - (input_data[next_vertex]-decp_data[index]))/2.0;
            double d =  (bound-(input_data[index]-decp_data[index]))/2.0;
            
            
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                de_direction_ds[index]=or_direction_ds[index];
                return 0;
            }
            
            if(diff>=1e-16 or d>=1e-16){
                if(decp_data[index]==decp_data[next_vertex]){
                    
                    
                        while(abs(input_data[next_vertex]-decp_data[index]-d)>bound and d>=2e-16){
                            d/=2;
                        }
                        
                        if(abs(input_data[index]-decp_data[index]-d)<=bound){
                            decp_data[index]+=d;
                        }
                    
                    
                    
                    
                }
                else{
                    if(decp_data[index]<=decp_data[next_vertex]){
                        
                            while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
                                    diff/=2;
                            }
                            
                            if (abs(input_data[next_vertex]-decp_data[index]+d)<=bound and decp_data[index]<=decp_data[next_vertex]){
                                decp_data[next_vertex] = decp_data[index]-diff;
                            }
                            
                            
                        
                        
                };

                }
                
                

                
            }

            else{
                
                if(decp_data[index]<=decp_data[next_vertex]){
                    if(abs(input_data[index]-(input_data[next_vertex] + bound + decp_data[index])/2.0)<=bound){
                        decp_data[index] = (input_data[next_vertex] + bound + decp_data[index])/2.0;
                    }
                    else{
                        decp_data[index] = input_data[index] + bound;
                    }
                }
            }
            

            
            
            

            
        
        }
    
        else{
            
            int largest_index = from_direction_to_index(index,de_direction_ds[index]);
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            double d = (bound-(input_data[largest_index]-decp_data[index]))/2.0;
            
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                de_direction_ds[index] = -1;
                return 0;
            }
            
            if (diff>=1e-16){
                if (decp_data[index]>=decp_data[largest_index]){
                    while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
                        diff/=2;
                    }
                    
                    
                    if(abs(input_data[index]-decp_data[index]+diff)<=bound){
                        decp_data[index] -= diff;
                    }
                    
                    
                }                    
            }
            
                    
            else{
                if (decp_data[index]>=decp_data[largest_index]){
                    decp_data[index] = input_data[index] - bound;
                }   
    
            }


               
        }

        
    }    
    return 0;
};
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
            // if(wrong_index_as.size()==7 and wrong_index_ds.size()==0){
            //     cout<<de_direction_as[cur]<<","<<cur<<endl;
            //     cout<<or_direction_as[cur]<<","<<cur<<endl;
            //     cout<<decp_data[false_index]<<","<<false_index<<endl;
            //     cout<<decp_data[true_index]<<","<<true_index<<endl;
            // }
            if(decp_data[false_index]<decp_data[true_index]){
                
                de_direction_as[cur]=or_direction_as[cur];
                
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
                        decp_data[true_index] = decp_data[false_index] + diff;
                    }
                    if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
                        
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
int get_wrong_index_cp(){
    
    return 0;
};
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
bool compareIndices(int i1, int i2, const std::vector<double>& data) {
    if (data[i1] == data[i2]) {
        return i1<i2; // 注意这里是‘>’
    }
    
    return data[i1] < data[i2];
}

void get_false_criticle_points(){
    count_max=0;
    count_min=0;

    count_f_max=0;
    count_f_min=0;

    #pragma omp parallel for
    for (auto i = 0; i < size2; i ++) {
            bool is_maxima = true;
            bool is_minima = true;

            for (int j : adjacency[i]) {
                if (decp_data[j] > decp_data[i]) {
                    is_maxima = false;
                    break;
                }
                else if(decp_data[j] == decp_data[i] and j>i){
                    is_maxima = false;
                    break;
                }
            }
            for (int j : adjacency[i]) {
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
extern void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,int width, int height, std::vector<int> *lowGradientIndices, double bound,float &datatransfer,float &finddirection);
// extern void update_de_direction(std::vector<int> *c,std::vector<int> *d);
extern void fix_process(std::vector<int> *c,std::vector<int> *d, std::vector<double> *decp_data1, float &datatransfer, float &finddirection, float &getfcp, float &fixtime_cp);
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

int main(int argc, char** argv){

    
    omp_set_num_threads(44);
    std::string dimension = argv[1];
    double range = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int mode = std::stoi(argv[4]);
    double target_br;
    float datatransfer = 0.0;
    float mappath_path = 0.0;
    double getfpath = 0.0;
    double fixtime_path = 0.0;
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
            iss >> height >> delimiter) {
            std::cout << "Filename: " << filename << std::endl;
            std::cout << "Width: " << width << std::endl;
            std::cout << "Height: " << height << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for filename" << std::endl;
    }
    // std::string fix_path = filename+id+".bin";
    inputfilename = "/global/homes/y/yuxiaoli/msz/experiment_data/"+filename+".bin";
    input_data = getdata2(inputfilename);
    auto min_it = std::min_element(input_data.begin(), input_data.end());
    auto max_it = std::max_element(input_data.begin(), input_data.end());
    double minValue = *min_it;
    double maxValue = *max_it;
    bound = (maxValue-minValue)*range;
    std::ostringstream stream;
    // 设置精度为最大可能的双精度浮点数有效数字
    // stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << std::defaultfloat << bound;  // 使用默认的浮点表示
    std::string valueStr = stream.str();
    cout<<bound<<","<<std::to_string(bound)<<endl;
    // exit(0);
    size2 = input_data.size();

    std::string cpfilename = "/pscratch/sd/y/yuxiaoli/compressed_data/compressed_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".sz";
    std::string decpfilename = "/pscratch/sd/y/yuxiaoli/decompressed_data/decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string fix_path = "/pscratch/sd/y/yuxiaoli/decompressed_data/fixed_decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string command;
    cout<<decpfilename<<endl;
    int result;
    


    if(compressor_id=="sz3"){
        // cout<<std::to_string(bound)<<endl;
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
        cpfilename = "/pscratch/sd/y/yuxiaoli/compressed_data/compressed_"+filename+"_"+std::to_string(bound)+".zfp";
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
    
    // cout<<"7391的值:"<<decp_data[7391]<<endl;
    // cout<<abs(decp_data[1031263]-input_data[1031263])<<endl;
    // cout<<bound<<endl;
    // exit(0);
    std::vector<double> decp_data_copy(decp_data);
    
    // cout<<"7391的值:"<<input_data[7391]<<endl;
    // for(int i;i<size2;i++){
    //     // if(input_data[i]!=decp_data[i]){
    //     //     cout<<i<<endl;
    //     // }
    //     cout<<input_data[i]<<endl;
    // }
    
    // width = 128;
    // height = 128;
    // depth = 128;
    
    or_direction_as.resize(size2);
    or_direction_ds.resize(size2);
    de_direction_as.resize(size2);
    de_direction_ds.resize(size2);
    or_label.resize(size2*2, -1);
    dec_label.resize(size2*2, -1);
    lowGradientIndices = find_low();
    lowGradientIndices.resize(size2,0);
    adjacency =  _compute_adjacency();
    // size2 = width*height;
    
    // cout<<lowGradientIndices.size()<<endl;
    all_max.resize(size2);
    all_min.resize(size2);
    // all_d_max.resize(size2);
    // all_d_min.resize(size2);
    std::random_device rd;  // 随机数生成器
    std::mt19937 gen(rd()); // 以随机数种子初始化生成器
    // std::string result = "{";
    // for (const auto& item : reverse_direction_mapping) {
    //     result+=std::to_string(item.first);
    //     result+=",";
    //     result+=std::to_string(item.second.first);
    //     result+=",";
    //     result+=std::to_string(item.second.second);
    //     result+=",";
    //     // std::cout << "Key: " << item.first
    //     //           << " Value: (" << item.second.first << ", " << item.second.second << ")\n";
    // }
    // result+="}";
    // cout<<result<<endl; 
    
    // 创建一个均匀分布的随机数生成器
    
    
    
    
    int cnt=0;
    
    // printf("%d %d\n",width,height);
    // cout<<ordered_input_data.size()<<endl;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    // #pragma omp parallel for
    // for (int i=0;i<size2;++i){
    //     // if(i%100000==0) cout<<i/100000<<endl;
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
        
        
       
        

    // };
    // cout<<"dahk"<<endl;

    
    
    
    // #pragma omp parallel for
    
    // for (int i=0;i<size2;++i){
        
    //         de_direction_as[i] = find_direction(i,decp_data,0);
        
    //         de_direction_ds[i] = find_direction(i,decp_data,1);
        
    // };
    
    // cout<<duration.count()<<"s"<<endl;
    // exit(0);
    // searchdirection_time+=duration.count();
    std::cout.precision(std::numeric_limits<double>::max_digits10);


    
    // or_label = mappath(or_mesh,or_direction_as,or_direction_ds);
    
    
    
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
    
    
    // getfcp+=d.count();
    // cout<<"错误cp:"<<count_f_max<<","<<count_f_min<<endl;
    // // return 0;
    // fix_process(decp_data);
    
    // cout<<"第一次修复开始"<<endl;
    
    // // std::uniform_int_distribution<> distrib(0, wrong_maxi_cp.size() - 1);
    // // std::uniform_int_distribution<> distrib1(0, wrong_min_cp.size() - 1);
    // // int randomIndex = distrib(gen);
    // // int randomIndex1 = distrib1(gen);
    
    
    // while (count_f_max>0 or count_f_min>0){
    //     cpite +=1;
    //     start5 = std::chrono::high_resolution_clock::now();
    //     cout<<"错误cp:"<<count_f_max<<","<<count_f_min<<endl;
    //     // auto temp = 0;
        
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
    //     // #pragma omp parallel for
    //     // for (int i=0;i<size2;++i){
    //     //     // auto it = __gnu_parallel::find(all_d_max.begin(), all_d_max.end(), i);
    //     //     // if ( it == all_d_max.end()){
    //     //         de_direction_as[i] = find_direction(i,decp_data,0);
    //     //     // }
    //     //     // else{            
    //     //         // de_direction_as[i] = -1;
    //     //     // };
    //     //     // auto it1 = __gnu_parallel::find(all_d_min.begin(), all_d_min.end(), i);
    //     //     // if (it1 == all_d_min.end()){
    //     //         de_direction_ds[i] = find_direction(i,decp_data,1);
    //     //     // }
    //     //     // else{
                
    //     //         // de_direction_ds[i] = -1;
    //     //     // };
            
            
    //     // };
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
    auto start = std::chrono::high_resolution_clock::now();
    // init_or_data(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f,size2);
    // cout<<"7391的值:"<<decp_data[7391]<<endl;
    init_inputdata(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, width, height, dev_g, bound,datatransfer,finddirection);
    // update_de_direction(dev_c,dev_d);
    // cout<<input_data[1949]<<", "<<decp_data[1949]<<endl;
    // for(int i:adjacency[1949]){
    //     cout<<i<<", "<<input_data[i]<<", "<<decp_data[i]<<endl;
    // }
    // cout<<or_direction_as[1949]<<endl;
    // exit(0);
    // cout<<"原来的:"<<input_data[1949]<<", "<<decp_data[1949]<<endl;
    
    
    
    // cout<<"djal"<<endl;
    float calculatermappath = 0.0;
    cout<<"map开始"<<endl;
    mappath1(dev_m, dev_a, dev_b,finddirection, mappath_path, datatransfer, 1);
    cout<<"map结束"<<endl;
    mappath1(dev_q, dev_c, dev_d,finddirection, mappath_path, datatransfer);
    cout<<"map结束"<<endl;
    size_t size = or_label.size();
    std::ofstream outFile11("/pscratch/sd/y/yuxiaoli/label/or_label_"+filename+".bin", std::ios::binary);

    if (!outFile11.is_open()) {
        throw std::runtime_error("Unable to open file for writing.");
    }
    outFile11.write(reinterpret_cast<const char*>(&size), sizeof(size));

    if (size > 0) {
        outFile11.write(reinterpret_cast<const char*>(or_label.data()), size * sizeof(int));
    }

    outFile11.close();
    
    std::ofstream outFile12("/pscratch/sd/y/yuxiaoli/label/dec_label_"+filename+'_'+compressor_id+'_'+std::to_string(bound)+".bin", std::ios::binary);
    cout<<"存入了: "<<"/pscratch/sd/y/yuxiaoli/label/dec_label_"+filename+'_'+compressor_id+'_'+std::to_string(bound)+".bin"<<endl;
    // // 写入大小
    outFile12.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // 写入vector的数据
    cout<<size<<endl;
    if (size > 0) {
        outFile12.write(reinterpret_cast<const char*>(dec_label.data()), size * sizeof(int));
    }
    cout<<"写完了"<<endl;
    
    // exit(0);
    // 关闭文件
    outFile12.close();
    // get_false_criticle_points();
    // here
    // cout<<"7391的值:"<<decp_data[7391]<<endl;
    // 这里
    auto startt = std::chrono::high_resolution_clock::now();
    double right_labeled_ratio = 1-get_wrong_index_path();
    // cout<<right_labeled_ratio<<endl;
    // exit(0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - startt;
    
    getfpath+=duration.count();
    // std::vector<double>* dev_f = &decp_data;
    std::vector<double> decp_data_c1(decp_data); 

    std::vector<double>* dev_k = &decp_data_c1;
    // float counter_getfcp = 0.0;
    float counter_fixtime_cp = 0.0;
    // cout<<"djaldjas"<<endl;
    // cout<<"7391的值:"<<decp_data[7391]<<endl;
    // exit(0);
    fix_process(dev_c,dev_d,dev_f,datatransfer, finddirection, getfcp, fixtime_cp);
    // cout<<duration1.count()<<"s"<<endl;
    // exit(0);
    // init_or_data(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f,size2);
        
    // cout<<"第一次修复结束"<<endl;
    // // exit(0);
    // end1 = std::chrono::high_resolution_clock::now();
    // end2 = std::chrono::high_resolution_clock::now();
    // duration1 = end1 - start1;
    // //  fixtime += duration1.count();
    // duration1 = end2 - start2;
    // fixtime_cp += duration1.count();
    startt = std::chrono::high_resolution_clock::now();
    // start2 = std::chrono::high_resolution_clock::now();
        
        
    // };
    // de_mesh.get_criticle_points();
    
        
    
    // #pragma omp parallel for
    // for (int i=0;i<size2;++i){
        
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

    
    dev_a = &or_direction_as;
    dev_b = &or_direction_ds;
    dev_c = &de_direction_as;
    dev_d = &de_direction_ds;
    // for(int i=0;i<1000;i++){
    mappath1(dev_q, dev_c, dev_d, finddirection, mappath_path, datatransfer);
    // }
    
    // exit(0);
    // cout<<"djaldjas"<<endl;
    
    // dec_label = mappath(de_mesh,de_direction_as,de_direction_ds);
    startt = std::chrono::high_resolution_clock::now();
    get_wrong_index_path();
    // get_false_criticle_points();
    
    
    
    end = std::chrono::high_resolution_clock::now();
    duration = end - startt;
    getfpath+=duration.count();
    // std::ofstream outFile1("or_label"+filename+".bin", std::ios::out | std::ios::binary);

    // if (!outFile1) {
    //     std::cerr << "Cannot open the output file." << std::endl;
    //     return 1;
    // }

    // // 遍历二维 vector 并将每个整数写入到文件中
    // for (int num : or_label) {
    //     outFile1.write(reinterpret_cast<const char*>(&num), sizeof(num));
    // }

    // // 关闭文件
    // outFile1.close();
    // exit(0);
    
    while (wrong_index_as.size()>0 or wrong_index_ds.size()>0 or count_f_max>0 or count_f_min>0){
        // cout<<"修复path:"<<wrong_index_as.size()<<", "<<wrong_index_ds.size()<<endl;
        ite+=1;
        // fixtime_path = 0;
        // searchdirection_time = 0;
        // searchtime = 0;
        // getfcp = 0;
        // get_path = 0; 
        // whole = 0;
        // std::vector<float> temp_time;
        // std::vector<float> temp_time_ratio;
        // cout<<"修复路径: "<<wrong_index_as.size()<<","<<wrong_index_ds.size()<<endl;
        
        // if(wrong_index_as.size()==0 and wrong_index_ds.size()==2){
        //     for(int i:wrong_index_ds){
        //         cout<<i<<", "<<or_label[i*2]<<", "<<dec_label[i*2]<<endl;
        //     }
        // }
        // exit(0);
        startt = std::chrono::high_resolution_clock::now();
        
        // start3 = std::chrono::high_resolution_clock::now();
        for(int i =0;i< wrong_index_as.size();i++){
            int j = wrong_index_as[i];
            if(lowGradientIndices[j] == 1)
            {
                continue;
                }
            fixpath(j,0);
        };
        // cout<<"danfjka"<<endl;
        for(int i =0;i< wrong_index_ds.size();i++){
            int j = wrong_index_ds[i];
            if(lowGradientIndices[j] == 1)
            {
                continue;
                }
            fixpath(j,1);
        };
        end = std::chrono::high_resolution_clock::now();
        duration = end - startt;
        fixtime_path+=duration.count();
        // cout<<"danfjka"<<endl;
        // end3 = std::chrono::high_resolution_clock::now();
        // duration1 = end3 - start3;
        // fixtime_path += duration1.count();
        
        
        
        
        // start2 = std::chrono::high_resolution_clock::now();
        // #pragma omp parallel for
        // for (int i=0;i<size2;++i){
        //         de_direction_as[i] = find_direction(i,decp_data,0);
        //         de_direction_ds[i] = find_direction(i,decp_data,1);
        // };
        // end2 = std::chrono::high_resolution_clock::now();
        // duration1 = end2 - start2;
        // searchdirection_time+=duration1.count();
        
   
        
        // start2 = std::chrono::high_resolution_clock::now();
        
        int cpite = 0;
        
        // while(count_f_max>0 or count_f_min>0){
        
        //     cpite +=1;
        //     start5 = std::chrono::high_resolution_clock::now();
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
            
            
        
        
        //     start2 = std::chrono::high_resolution_clock::now();
            
        //     get_false_criticle_points();
        //     end2 = std::chrono::high_resolution_clock::now();
        //     duration1 = end2 - start2;
        //     getfcp += duration1.count();
        

        // }
        // start2 = std::chrono::high_resolution_clock::now();
        
        dev_c = &de_direction_as;
        dev_d = &de_direction_ds;
        
        dev_f = &decp_data;
        
        fix_process(dev_c,dev_d,dev_f,datatransfer, finddirection, getfcp, fixtime_cp);
        // end2 = std::chrono::high_resolution_clock::now();
        // duration1 = end2 - start2;
        // getfcp += duration1.count();
        // start2 = std::chrono::high_resolution_clock::now();
        
        
        
        // end2 = std::chrono::high_resolution_clock::now();
        // duration1 = end2 - start2;
        
        // searchdirection_time+=duration1.count();
        
        // start2 = std::chrono::high_resolution_clock::now();
        
        // cout<<de_direction_ds[3030]<<endl;
        dev_q = &dec_label;
        mappath1(dev_q, dev_c, dev_d,finddirection, mappath_path, datatransfer);
        // end2 = std::chrono::high_resolution_clock::now();
        // duration1 = end2 - start2;
        startt = std::chrono::high_resolution_clock::now();
        // searchtime+=duration1.count();
        // start2 = std::chrono::high_resolution_clock::now();
        
        
        
        double r = get_wrong_index_path();
        // end2 = std::chrono::high_resolution_clock::now();
        // duration1 = end2-startt;
        // whole += duration1.count();
        end = std::chrono::high_resolution_clock::now();
        duration = end - startt;
        getfpath+=duration.count();
        // duration1 = end1-start1;
        // get_path+=duration1.count();
        // // cout<<"dfahlhql: "<<whole<<endl;
        // temp_time.push_back(getfcp);
        // temp_time_ratio.push_back(getfcp/whole);
        // temp_time.push_back(fixtime_cp);
        // temp_time_ratio.push_back(fixtime_cp/whole);
        // temp_time.push_back(fixtime_path);
        // temp_time_ratio.push_back(fixtime_path/whole);
        // temp_time.push_back(searchdirection_time);
        // temp_time_ratio.push_back(searchdirection_time/whole);
        // temp_time.push_back(searchtime);
        // temp_time_ratio.push_back(searchtime/whole);
        // temp_time.push_back(get_path);
        // temp_time_ratio.push_back(get_path/whole);
        // record1.push_back(temp_time);
        // record_ratio.push_back(temp_time_ratio);
    };

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end - start;
    cout<<"duration: "<<duration1.count()*1000<<endl;
    cout<<"data_transfer:"<<datatransfer<<endl;
    cout<<"find_direction:"<<finddirection<<endl;
    cout<<"getfcp:"<<getfcp<<endl;
    cout<<"mappath_path:"<<mappath_path<<endl;
    cout<<"getfpath:"<<getfpath*1000<<endl;
    cout<<"fixfcp:"<<fixtime_cp<<endl;
    cout<<"fixpath:"<<fixtime_path*1000<<endl;
    std::ofstream outFile13("/pscratch/sd/y/yuxiaoli/label/fixed_dec_label_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin");
    if (!outFile13.is_open()) {
        throw std::runtime_error("Unable to open file for writing.");
    }

    // 获取vector的大小
    
    
    // 写入大小
    outFile13.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // 写入vector的数据
    if (size > 0) {
        outFile13.write(reinterpret_cast<const char*>(&dec_label[0]), size * sizeof(int));
    }

    // 关闭文件
    outFile13.close();
    // std::cout << "fixtime: " << fixtime << " seconds" << std::endl;
    // std::cout << "get_fcp: " << getfcp/duration.count() << " seconds" << std::endl;
    // std::cout << "fixtime_cp: " << fixtime_cp/duration.count() << " seconds" << std::endl;
    // std::cout << "fixtime_path: " << fixtime_path/duration.count() << " seconds" << std::endl;
    // std::cout << "searchtime: " << searchtime << " seconds" << std::endl;
    // std::cout << "finddirection: " << searchdirection_time/duration.count() << " seconds" << std::endl;
    // std::cout << "getfpath:" << get_path/duration.count() << std::endl;
    // std::cout << "iteration number:" << ite << std::endl;
    // std::ofstream file2("dec_label_iteration内"+std::to_string(width)+".txt");
    // result = "";
    // for (const auto& p : dec_label) {
    //     result+="[";
    //     result+=std::to_string(p.first)+","+std::to_string(p.second);
    //     result+="],";
    // }
    // file2 << result << "\n";
    // file2 << "#"<<",";
    
    // result = "[";
    // for( int i:de_mesh.maxi){
        
    //     result+=std::to_string(i);
    //     result+=",";
    // }
    // result+="]";
    // file2 << result;
    // file2 << "\n";
    // file2 << "#"<<",";
    // result = "[";
    // for( int i:de_mesh.mini){
        
    //     result+=std::to_string(i);
    //     result+=",";
    // }
    // result+="]";
    // file2 << result;

    // // // 关闭文件
    // file2.close();
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
    // std::ofstream file("cppfixed1_decomp"+std::to_string(width)+".csv");
    
    

    // // file << std::setprecision(std::numeric_limits<double>::max_digits10);
    // for (int i = 0; i < ordered_decp_data.size(); ++i) {
    //     file << de_mesh.values[i];
        
    //     if (i < decp_data.size() - 1) 
    //         file << ", ";
    
    // }

   std::ofstream outFile(fix_path);
    // std::ofstream outFile("values.bin", std::ios::binary | std::ios::out);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(decp_data.data()), decp_data.size() * sizeof(double));
    }
    outFile.close();
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
    double compressed_dataSize = fs::file_size(cpfilename);
    // cout<<compressed_indexSize<<", "<<original_indexSize<<", "<<original_indexSize/compressed_indexSize<<endl;
    // cout<<compressed_editSize<<", "<<original_editSize<<", "<<original_editSize/compressed_editSize<<endl;
    // cout<<compressed_dataSize<<", "<<original_dataSize<<", "<<original_dataSize/compressed_dataSize<<endl;
    double overall_ratio = (original_indexSize+original_editSize+original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    double bitRate = 64/overall_ratio; 
    
    


    double psnr = calculatePSNR(input_data, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data, decp_data, maxValue-minValue);
    cout<<psnr<<", "<<fixed_psnr<<endl;
    cout<<"right: "<<right_labeled_ratio<<endl;
    cout<<"relative range: "<<range<<endl;
    
    std::ofstream outFile3("../result/result_"+filename+"_"+compressor_id+"tmp2_0_1.txt", std::ios::app);

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

    outFile3 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << "\n" << std::endl;
    // 关闭文件
    outFile3.close();


    std::cout << "Variables have been appended to output.txt" << std::endl;
    
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
    // // std::cout<<result<<std::endl;
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
    return 0;
}


